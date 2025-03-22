from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from fla.layers.delta_net import DeltaNet
from flazoo.models.attentions import VisionAttention
from ..utils import get_2d_sincos_pos_embed, modulate
from ..utils import Gen2DTimestepEmbedder
from ..utils import Gen2DPatchEmbed, Gen2DLabelEmbedder
from flazoo.models.utils import prepare_hidden_states_for_scan, prepare_hidden_states_for_merge

class DeltaNetGen2DMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.mlp_dim, config.hidden_size),
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
        self.norm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = VisionAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                layer_idx=layer_idx
            )
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
                layer_idx=layer_idx
            )

        self.norm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.mlp = DeltaNetGen2DMLP(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x

class DeltaNetGen2DFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DeltaNetGen2DConfig:
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        qk_norm=False,
        **kwargs
    ):
        self.path_type = path_type
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_depth = encoder_depth
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.use_cfg = use_cfg
        self.z_dims = z_dims
        self.projector_dim = projector_dim
        self.qk_norm = qk_norm
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class DeltaNetForGen2D(nn.Module):
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
        self.y_embedder = Gen2DLabelEmbedder(config.num_classes, config.hidden_size, config.class_dropout_prob)
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, config.hidden_size), 
            requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            DeltaNetGen2DBlock(
                config.hidden_size, 
                config.num_heads, 
                mlp_ratio=config.mlp_ratio, 
                qk_norm=config.qk_norm
            ) for _ in range(config.depth)
        ])
        
        self.projectors = nn.ModuleList([
            DeltaNetGen2DProjectorMLP(config.hidden_size, config.projector_dim, z_dim) for z_dim in config.z_dims
        ])
        
        self.final_layer = DeltaNetGen2DFinalLayer(
            config.decoder_hidden_size, 
            config.patch_size, 
            self.out_channels
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
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        p = self.x_embedder.patch_size if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, return_logvar=False):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), 其中 T = H * W / patch_size ** 2
        N, T, D = x.shape

        t_embed = self.t_embedder(t)             # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                          # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if (i + 1) == self.encoder_depth:
                zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x, zs

