# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_linear_attn import LinearAttentionVisionConfig
from .modeling_linear_attn import (
    LinearAttentionVisionModel,
    LinearAttentionForImageClassification,
    LinearAttentionForMaskedImageModeling,
)

AutoConfig.register(LinearAttentionVisionConfig.model_type, LinearAttentionVisionConfig)
AutoModelForImageClassification.register(
    LinearAttentionVisionConfig, LinearAttentionForImageClassification
)
AutoModelForMaskedImageModeling.register(
    LinearAttentionVisionConfig, LinearAttentionForMaskedImageModeling
)
AutoModel.register(LinearAttentionVisionConfig, LinearAttentionVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_linear_attn import LinearAttentionVideoConfig
from .modeling_linear_attn import (
    LinearAttentionVideoModel,
    LinearAttentionForVideoClassification,
    LinearAttentionForVideoPreTraining,
)

AutoConfig.register(LinearAttentionVideoConfig.model_type, LinearAttentionVideoConfig)
AutoModel.register(LinearAttentionVideoConfig, LinearAttentionVideoModel)
AutoModelForVideoClassification.register(
    LinearAttentionVideoConfig, LinearAttentionForVideoClassification
)
AutoModelForPreTraining.register(
    LinearAttentionVideoConfig, LinearAttentionForVideoPreTraining
)

__all__ = [
    "LinearAttentionVisionModel",
    "LinearAttentionForImageClassification",
    "LinearAttentionForMaskedImageModeling",
    "LinearAttentionVisionConfig",
    "LinearAttentionVideoModel",
    "LinearAttentionForVideoClassification",
    "LinearAttentionForVideoPreTraining",
    "LinearAttentionVideoConfig",
]
