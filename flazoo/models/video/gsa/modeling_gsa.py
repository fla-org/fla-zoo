# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (ImageClassifierOutput,
                                           BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flazoo.models.utils import VAttention
from fla.layers.gsa import GatedSlotAttention
from .configuration_gsa import GSAVideoConfig
from fla.models.utils import Cache
from fla.modules import (FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss,
                         RMSNorm)
from fla.modules.activations import swiglu_linear
from fla.modules.layernorm import rms_norm_linear
from flazoo.models.utils import prepare_hidden_states_for_scan, prepare_hidden_states_for_merge
from ..utils import VideoEmbeddings, VideoDecoderOutput, VideoForPreTrainingOutput, get_sinusoid_encoding_table
from copy import deepcopy
from ...scan import RandomScanWithReorder

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

class GSAVideoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, x):
        return self.net(x)

class GSAVideoBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = VAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedSlotAttention(
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                num_slots=config.num_slots,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                feature_map=config.feature_map,
                use_output_gate=config.use_output_gate,
                use_norm=config.use_norm,
                gate_fn=config.hidden_act,
                gate_logit_normalizer=config.gate_logit_normalizer,
                elementwise_affine=config.elementwise_affine,
                norm_first=config.norm_first,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx
            )
            
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
        self.mlp = GSAVideoMLP(config)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.train_scan_type = 'uni-scan'
            self.test_scan_type = 'uni-scan'
        else:
            self.train_scan_type = config.train_scan_type
            self.test_scan_type = config.test_scan_type


    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: bool = False,
        **kwargs: Unpack[Dict]
    ):
        residual = hidden_states
        
        hidden_states = self.ln_1(hidden_states)
        hidden_states = prepare_hidden_states_for_scan(hidden_states, self.train_scan_type)

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        hidden_states = prepare_hidden_states_for_merge(hidden_states, self.train_scan_type)

        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        outputs = (hidden_states, attentions, past_key_values)

        return outputs

class GSAVideoEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [GSAVideoBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict=True,
        **kwargs
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, _ = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attentions, _ = block(
                    hidden_states,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

class GSAVideoPreTrainedModel(PreTrainedModel):
    config_class = GSAVideoConfig
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class GSAVideoModel(GSAVideoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = VideoEmbeddings(config)
        self.encoder = GSAVideoEncoder(config)

        self.post_init()

    def forward(
        self,
        pixel_values,
        bool_masked_pos=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict=None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values, bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class GSAVideoDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()

        decoder_num_labels = config.num_channels * config.tubelet_size * config.patch_size**2

        # Initialize decoder-specific configuration
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_heads = config.decoder_num_heads
        decoder_config.mlp_dim = config.decoder_mlp_dim

        self.decoder_blocks = nn.ModuleList(
            [GSAVideoBlock(decoder_config, layer_idx) for layer_idx in range(decoder_config.num_hidden_layers)]
        )

        self.norm = nn.LayerNorm(config.decoder_hidden_size)
        self.head = nn.Linear(config.decoder_hidden_size, decoder_num_labels) if decoder_num_labels > 0 else nn.Identity()

        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states,
        return_token_num,
        output_attentions=False,
        output_hidden_states=False,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict=True,
        **kwargs
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, block in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, _ = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs
                )
            else:
                hidden_states, attentions, _ = block(
                    hidden_states,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs
                )

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_token_num > 0:
            hidden_states = hidden_states[:, -return_token_num:]

        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)

        return VideoDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

class GSAForVideoPreTraining(GSAVideoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.backbone = GSAVideoModel(config)
        
        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.position_embeddings = get_sinusoid_encoding_table(
            self.backbone.embeddings.num_patches, config.decoder_hidden_size
        )

        self.decoder = GSAVideoDecoder(config, num_patches=self.backbone.embeddings.num_patches)

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, VideoForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.encoder_to_decoder(sequence_output)
        
        batch_size, seq_len, num_channels = sequence_output.shape

        expanded_position_embeddings = self.position_embeddings.expand(batch_size, -1, -1).type_as(pixel_values)
        expanded_position_embeddings = expanded_position_embeddings.to(pixel_values.device).clone().detach()
        
        pos_emb_visible = expanded_position_embeddings[~bool_masked_pos].reshape(batch_size, -1, num_channels)
        pos_emb_mask = expanded_position_embeddings[bool_masked_pos].reshape(batch_size, -1, num_channels)

        x_full = torch.cat([sequence_output + pos_emb_visible, self.mask_token + pos_emb_mask], dim=1)

        decoder_outputs = self.decoder(x_full, pos_emb_mask.shape[1])
        logits = decoder_outputs.logits

        loss = None
        # below taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L730
        with torch.no_grad():
            # calculate the labels to be predicted
            if self.config.num_channels != 3:
                # Can't unnormalize with default means/stds
                frames = pixel_values
            else:
                # first, unnormalize the frames
                device = pixel_values.device
                dtype = pixel_values.dtype
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device=device, dtype=dtype)[None, None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device=device, dtype=dtype)[None, None, :, None, None]
                frames = pixel_values * std + mean  # in [0, 1]

            batch_size, time, num_channels, height, width = frames.shape
            tubelet_size, patch_size = self.config.tubelet_size, self.config.patch_size
            if self.config.norm_pix_loss:
                # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
                frames = frames.view(
                    batch_size,
                    time // tubelet_size,
                    tubelet_size,
                    num_channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                )
                # step 2: move dimensions to concatenate:
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                # step 3: concatenate:
                frames = frames.view(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size,
                    num_channels,
                )
                # step 4: normalize. The authors find that the mean is about 0.48 and standard deviation is about 0.08.
                frames_norm = (frames - frames.mean(dim=-2, keepdim=True)) / (
                    frames.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                )
                # step 5: reshape to (batch_size, T//ts * H//ps * W//ps, ts * ps * ps * C)
                videos_patch = frames_norm.view(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size * num_channels,
                )
            else:
                if self.config.num_channels != 3:
                    raise ValueError(
                        "Can't unnormalize non-RGB images. Consider setting config.norm_pix_loss to False."
                    )
                # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
                frames = frames.view(
                    batch_size,
                    time // tubelet_size,
                    tubelet_size,
                    num_channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                )
                # step 2: move dimensions to concatenate: (batch_size, T//ts, H//ps, W//ps, ts, ps, ps, C)
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                # step 3: concatenate
                videos_patch = frames.view(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size * num_channels,
                )

            batch_size, _, num_channels = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(batch_size, -1, num_channels)

        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return VideoForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class GSAForVideoClassification(GSAVideoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_classes
        self.backbone = GSAVideoModel(config)

        # Classifier head
        self.fc_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes) if config.num_classes > 0 else nn.Identity()

        self.post_init()

    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

