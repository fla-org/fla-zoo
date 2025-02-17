# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_linear_attn import LinearAttentionVideoConfig
from .modeling_linear_attn import(LinearAttentionVideoModel, LinearAttentionForVideoClassification, LinearAttentionForVideoPreTraining)

AutoConfig.register(LinearAttentionVideoConfig.model_type, LinearAttentionVideoConfig)
AutoModel.register(LinearAttentionVideoConfig, LinearAttentionVideoModel)
AutoModelForVideoClassification.register(LinearAttentionVideoConfig, LinearAttentionForVideoClassification)
AutoModelForPreTraining.register(LinearAttentionVideoConfig, LinearAttentionForVideoPreTraining)

__all__ = ['LinearAttentionVideoConfig', 'LinearAttentionVideoModel', 'LinearAttentionForVideoClassification', 'LinearAttentionForVideoPreTraining']