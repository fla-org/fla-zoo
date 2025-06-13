from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from fla.layers.delta_net import DeltaNet
from flazoo.models.attentions import FullAttention
from ..utils import get_2d_sincos_pos_embed, modulate
from ..embeddings import Gen2DTimestepEmbedder
from fla.modules.layernorm import rms_norm_linear, LayerNorm
from flazoo.models.attentions import get_attn
from ..embeddings import Gen2DLabelEmbedder
from timm.layers import PatchEmbed as Gen2DPatchEmbed
from flazoo.models.utils import (
    prepare_hidden_states_for_scan,
    prepare_hidden_states_for_merge,
)
from .configuration_delta_net import DeltaNetGen2DConfig


class DeltaNetGen2DMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.channel_mixer_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.channel_mixer_dim, config.hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class DeltaNetGen2DProjectorMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.projector_dim),
            nn.SiLU(),
            nn.Linear(config.projector_dim, config.z_dim),
            nn.SiLU(),
            nn.Linear(config.z_dim, config.hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class DeltaNetGen2DBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps
        )
        if config.attn is not None and layer_idx in config.attn["layers"]:
            self.attn = get_attn(config, layer_idx)
        else:
            self.attn = DeltaNet(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                use_gate=config.use_gate,
                use_beta=config.use_beta,
                use_short_conv=config.use_short_conv,
                use_output_norm=config.use_output_norm,
                conv_size=config.conv_size,
                qk_norm=config.qk_norm,
                qk_activation=config.qk_activation,
                norm_first=config.norm_first,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )

        self.norm2 = LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps
        )
        self.channel_mixer = DeltaNetGen2DMLP(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_channel_mixer,
            scale_channel_mixer,
            gate_channel_mixer,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn)
        )
        x = x + gate_channel_mixer.unsqueeze(1) * self.channel_mixer(
            modulate(self.norm2(x), shift_channel_mixer, scale_channel_mixer)
        )

        return x


class DeltaNetGen2DFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DeltaNetGen2DPreTrainedModel(PreTrainedModel):
    config_class =  DeltaNetGen2DConfig
    base_model_prefix = "deltanet_gen2d"
    supports_gradient_checkpointing = True


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DeltaNetForGen2D(DeltaNetGen2DPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.path_type = config.path_type
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads
        self.use_cfg = config.use_cfg
        self.num_classes = config.num_classes
        self.z_dims = config.z_dims
        self.encoder_depth = config.encoder_depth

        self.x_embedder = Gen2DPatchEmbed(
            config.input_size, config.patch_size, config.in_channels, config.hidden_size
        )
        self.t_embedder = Gen2DTimestepEmbedder(config.hidden_size)
        self.y_embedder = Gen2DLabelEmbedder(
            config.num_classes, config.hidden_size, config.class_dropout_prob
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, config.hidden_size),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList(
            [
                DeltaNetGen2DBlock(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.projectors = nn.ModuleList(
            [DeltaNetGen2DProjectorMLP(config) for z_dim in config.z_dims]
        )

        self.final_layer = DeltaNetGen2DFinalLayer(
            config.decoder_hidden_size, config.patch_size, self.out_channels
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y, return_logvar=False):
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y  # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, T, D)
            if (i + 1) == self.encoder_depth:
                zs = [
                    projector(x.reshape(-1, D)).reshape(N, T, -1)
                    for projector in self.projectors
                ]
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x, zs
