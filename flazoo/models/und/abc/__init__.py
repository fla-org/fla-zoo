# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_abc import ABCVisionConfig
from .modeling_abc import (
    ABCVisionModel,
    ABCForImageClassification,
    ABCForMaskedImageModeling,
)

AutoConfig.register(ABCVisionConfig.model_type, ABCVisionConfig)
AutoModelForImageClassification.register(ABCVisionConfig, ABCForImageClassification)
AutoModelForMaskedImageModeling.register(ABCVisionConfig, ABCForMaskedImageModeling)
AutoModel.register(ABCVisionConfig, ABCVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_abc import ABCVideoConfig
from .modeling_abc import (
    ABCVideoModel,
    ABCForVideoClassification,
    ABCForVideoPreTraining,
)

AutoConfig.register(ABCVideoConfig.model_type, ABCVideoConfig)
AutoModel.register(ABCVideoConfig, ABCVideoModel)
AutoModelForVideoClassification.register(ABCVideoConfig, ABCForVideoClassification)
AutoModelForPreTraining.register(ABCVideoConfig, ABCForVideoPreTraining)

__all__ = [
    "ABCVisionModel",
    "ABCForImageClassification",
    "ABCForMaskedImageModeling",
    "ABCVisionConfig",
    "ABCVideoModel",
    "ABCForVideoClassification",
    "ABCForVideoPreTraining",
    "ABCVideoConfig",
]
