# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_transformer import TransformerVisionConfig
from .modeling_transformer import (
    TransformerVisionModel,
    TransformerForImageClassification,
    TransformerForMaskedImageModeling,
)

AutoConfig.register(TransformerVisionConfig.model_type, TransformerVisionConfig)
AutoModelForImageClassification.register(
    TransformerVisionConfig, TransformerForImageClassification
)
AutoModelForMaskedImageModeling.register(
    TransformerVisionConfig, TransformerForMaskedImageModeling
)
AutoModel.register(TransformerVisionConfig, TransformerVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_transformer import TransformerVideoConfig
from .modeling_transformer import (
    TransformerVideoModel,
    TransformerForVideoClassification,
    TransformerForVideoPreTraining,
)

AutoConfig.register(TransformerVideoConfig.model_type, TransformerVideoConfig)
AutoModel.register(TransformerVideoConfig, TransformerVideoModel)
AutoModelForVideoClassification.register(
    TransformerVideoConfig, TransformerForVideoClassification
)
AutoModelForPreTraining.register(TransformerVideoConfig, TransformerForVideoPreTraining)

__all__ = [
    "TransformerVisionModel",
    "TransformerForImageClassification",
    "TransformerForMaskedImageModeling",
    "TransformerVisionConfig",
    "TransformerVideoModel",
    "TransformerForVideoClassification",
    "TransformerForVideoPreTraining",
    "TransformerVideoConfig",
]
