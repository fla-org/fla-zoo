# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_gsa import GSAVisionConfig
from .modeling_gsa import (
    GSAVisionModel,
    GSAForImageClassification,
    GSAForMaskedImageModeling,
)

AutoConfig.register(GSAVisionConfig.model_type, GSAVisionConfig)
AutoModelForImageClassification.register(GSAVisionConfig, GSAForImageClassification)
AutoModelForMaskedImageModeling.register(GSAVisionConfig, GSAForMaskedImageModeling)
AutoModel.register(GSAVisionConfig, GSAVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_gsa import GSAVideoConfig
from .modeling_gsa import (
    GSAVideoModel,
    GSAForVideoClassification,
    GSAForVideoPreTraining,
)


AutoConfig.register(GSAVideoConfig.model_type, GSAVideoConfig)
AutoModel.register(GSAVideoConfig, GSAVideoModel)
AutoModelForVideoClassification.register(GSAVideoConfig, GSAForVideoClassification)
AutoModelForPreTraining.register(GSAVideoConfig, GSAForVideoPreTraining)

__all__ = [
    "GSAVisionModel",
    "GSAForImageClassification",
    "GSAForMaskedImageModeling",
    "GSAVisionConfig",
    "GSAVideoModel",
    "GSAForVideoClassification",
    "GSAForVideoPreTraining",
    "GSAVideoConfig",
]
