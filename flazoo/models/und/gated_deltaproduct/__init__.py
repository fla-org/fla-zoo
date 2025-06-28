# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_gated_deltaproduct import GatedDeltaProductVisionConfig
from .modeling_gated_deltaproduct import (
    GatedDeltaProductVisionModel,
    GatedDeltaProductForImageClassification,
    GatedDeltaProductForMaskedImageModeling,
)

AutoConfig.register(
    GatedDeltaProductVisionConfig.model_type, GatedDeltaProductVisionConfig
)
AutoModelForImageClassification.register(
    GatedDeltaProductVisionConfig, GatedDeltaProductForImageClassification
)
AutoModelForMaskedImageModeling.register(
    GatedDeltaProductVisionConfig, GatedDeltaProductForMaskedImageModeling
)
AutoModel.register(GatedDeltaProductVisionConfig, GatedDeltaProductVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_gated_deltaproduct import GatedDeltaProductVideoConfig
from .modeling_gated_deltaproduct import (
    GatedDeltaProductVideoModel,
    GatedDeltaProductForVideoClassification,
    GatedDeltaProductForVideoPreTraining,
)

AutoConfig.register(
    GatedDeltaProductVideoConfig.model_type, GatedDeltaProductVideoConfig
)
AutoModel.register(GatedDeltaProductVideoConfig, GatedDeltaProductVideoModel)
AutoModelForVideoClassification.register(
    GatedDeltaProductVideoConfig, GatedDeltaProductForVideoClassification
)
AutoModelForPreTraining.register(
    GatedDeltaProductVideoConfig, GatedDeltaProductForVideoPreTraining
)

__all__ = [
    "GatedDeltaProductVisionModel",
    "GatedDeltaProductForImageClassification",
    "GatedDeltaProductForMaskedImageModeling",
    "GatedDeltaProductVisionConfig",
    "GatedDeltaProductVideoModel",
    "GatedDeltaProductForVideoClassification",
    "GatedDeltaProductForVideoPreTraining",
    "GatedDeltaProductVideoConfig",
]
