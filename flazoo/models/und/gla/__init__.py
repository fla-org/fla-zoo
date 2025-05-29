# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_gla import GLAVisionConfig
from .modeling_gla import (
    GLAVisionModel,
    GLAForImageClassification,
    GLAForMaskedImageModeling,
)

AutoConfig.register(GLAVisionConfig.model_type, GLAVisionConfig)
AutoModelForImageClassification.register(GLAVisionConfig, GLAForImageClassification)
AutoModelForMaskedImageModeling.register(GLAVisionConfig, GLAForMaskedImageModeling)
AutoModel.register(GLAVisionConfig, GLAVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_gla import GLAVideoConfig
from .modeling_gla import (
    GLAVideoModel,
    GLAForVideoClassification,
    GLAForVideoPreTraining,
)


AutoConfig.register(GLAVideoConfig.model_type, GLAVideoConfig)
AutoModel.register(GLAVideoConfig, GLAVideoModel)
AutoModelForVideoClassification.register(GLAVideoConfig, GLAForVideoClassification)
AutoModelForPreTraining.register(GLAVideoConfig, GLAForVideoPreTraining)

__all__ = [
    "GLAVisionModel",
    "GLAForImageClassification",
    "GLAForMaskedImageModeling",
    "GLAVisionConfig",
    "GLAVideoModel",
    "GLAForVideoClassification",
    "GLAForVideoPreTraining",
    "GLAVideoConfig",
]
