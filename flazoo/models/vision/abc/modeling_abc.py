# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union, Unpack, Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (ImageClassifierOutput,
                                           MaskedImageModelingOutput,
                                           BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from fla.models.utils import Cache
from fla.layers.abc import ABCAttention
from flazoo.models.utils import VAttention
from .configuration_abc import ABCVisionConfig
from fla.modules import (FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss,
                         RMSNorm)
from flazoo.models.utils import prepare_hidden_states_for_scan, prepare_hidden_states_for_merge
from ..utils import ImageEmbeddings, Pooler
from ...scan import RandomScanWithReorder
logger = logging.get_logger(__name__)


class ABCVisionMLP(nn.Module):
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

class ABCVisionBlock(nn.Module):
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
            self.attn = ABCAttention(
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_slots=config.num_slots,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
                clamp_max=config.clamp_max,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx
            )
            
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
        self.mlp = ABCVisionMLP(config)

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
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor]]:
        residual = hidden_states

        # Pre-normalization if enabled
        if hasattr(self, 'ln_1'):
            hidden_states = self.ln_1(hidden_states)

        # Apply attention
        hidden_states = prepare_hidden_states_for_scan(hidden_states, self.train_scan_type, training=self.training)
        
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        
        hidden_states = prepare_hidden_states_for_merge(hidden_states, self.train_scan_type)

        # First residual connection
        hidden_states = residual + hidden_states
        residual = hidden_states

        # Pre-normalization for MLP if enabled 
        if hasattr(self, 'ln_2'):
            hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs

class ABCVisionPreTrainedModel(PreTrainedModel):
    config_class = ABCVisionConfig
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ImageEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


class ABCVisionEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            ABCVisionBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values = block(
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
            attentions=all_self_attentions,
        )

class ABCVisionModel(ABCVisionPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        self.embeddings = ImageEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ABCVisionEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = Pooler(config) if add_pooling_layer else None
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding)
        
        encoder_outputs = self.encoder(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class ABCForImageClassification(ABCVisionPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_classes
        self.backbone = ABCVisionModel(config, add_pooling_layer=True) # Here we should use mean pooling
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.init_weights()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output) # only use mean pooling

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
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

class ABCForMaskedImageModeling(ABCVisionPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = ABCVisionModel(config, add_pooling_layer=False, use_mask_token=True) 
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )
        self.init_weights()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )
        
        outputs = self.backbone(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )