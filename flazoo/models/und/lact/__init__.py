# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_lact import LaCTVisionConfig
from .modeling_lact import (
    LaCTVisionModel,
    LaCTForImageClassification,
    LaCTForMaskedImageModeling,
)

AutoConfig.register(LaCTVisionConfig.model_type, LaCTVisionConfig)
AutoModel.register(LaCTVisionConfig, LaCTVisionModel)
AutoModelForImageClassification.register(
    LaCTVisionConfig, LaCTForImageClassification
)
AutoModelForMaskedImageModeling.register(
    LaCTVisionConfig, LaCTForMaskedImageModeling
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_lact import LaCTVideoConfig
from .modeling_lact import (
    LaCTVideoModel,
    LaCTForVideoClassification,
    LaCTForVideoPreTraining,
)

AutoConfig.register(LaCTVideoConfig.model_type, LaCTVideoConfig)
AutoModel.register(LaCTVideoConfig, LaCTVideoModel)
AutoModelForVideoClassification.register(
    LaCTVideoConfig, LaCTForVideoClassification
)
AutoModelForPreTraining.register(LaCTVideoConfig, LaCTForVideoPreTraining)

__all__ = [
    "LaCTVisionModel",
    "LaCTForImageClassification",
    "LaCTForMaskedImageModeling",
    "LaCTVisionConfig",
    "LaCTVideoModel",
    "LaCTForVideoClassification",
    "LaCTForVideoPreTraining",
    "LaCTVideoConfig",
]
