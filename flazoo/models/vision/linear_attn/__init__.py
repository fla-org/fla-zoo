# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_linear_attn import \
    LinearAttentionVisionConfig
from .modeling_linear_attn import (
    LinearAttentionVisionModel, LinearAttentionForImageClassification, LinearAttentionForMaskedImageModeling)

AutoConfig.register(LinearAttentionVisionConfig.model_type, LinearAttentionVisionConfig)
AutoModelForImageClassification.register(LinearAttentionVisionConfig, LinearAttentionForImageClassification)
AutoModelForMaskedImageModeling.register(LinearAttentionVisionConfig, LinearAttentionForMaskedImageModeling)
AutoModel.register(LinearAttentionVisionConfig, LinearAttentionVisionModel)

__all__ = ['LinearAttentionVisionModel', 'LinearAttentionForImageClassification', 'LinearAttentionForMaskedImageModeling', 'LinearAttentionVisionConfig']
