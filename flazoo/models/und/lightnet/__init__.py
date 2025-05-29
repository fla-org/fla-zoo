# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_lightnet import LightNetVisionConfig
from .modeling_lightnet import (
    LightNetVisionModel,
    LightNetForImageClassification,
    LightNetForMaskedImageModeling,
)

AutoConfig.register(LightNetVisionConfig.model_type, LightNetVisionConfig)
AutoModel.register(LightNetVisionConfig, LightNetVisionModel)
AutoModelForImageClassification.register(
    LightNetVisionConfig, LightNetForImageClassification
)
AutoModelForMaskedImageModeling.register(
    LightNetVisionConfig, LightNetForMaskedImageModeling
)

__all__ = [
    "LightNetVisionModel",
    "LightNetForImageClassification",
    "LightNetForMaskedImageModeling",
    "LightNetVisionConfig",
]
