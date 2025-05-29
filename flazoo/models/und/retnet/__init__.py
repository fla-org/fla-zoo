# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_retnet import RetNetVisionConfig
from .modeling_retnet import (
    RetNetVisionModel,
    RetNetForImageClassification,
    RetNetForMaskedImageModeling,
)

AutoConfig.register(RetNetVisionConfig.model_type, RetNetVisionConfig)
AutoModel.register(RetNetVisionConfig, RetNetVisionModel)
AutoModelForImageClassification.register(
    RetNetVisionConfig, RetNetForImageClassification
)
AutoModelForMaskedImageModeling.register(
    RetNetVisionConfig, RetNetForMaskedImageModeling
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_retnet import RetNetVideoConfig
from .modeling_retnet import (
    RetNetVideoModel,
    RetNetForVideoClassification,
    RetNetForVideoPreTraining,
)

AutoConfig.register(RetNetVideoConfig.model_type, RetNetVideoConfig)
AutoModel.register(RetNetVideoConfig, RetNetVideoModel)
AutoModelForVideoClassification.register(
    RetNetVideoConfig, RetNetForVideoClassification
)
AutoModelForPreTraining.register(RetNetVideoConfig, RetNetForVideoPreTraining)

__all__ = [
    "RetNetVisionModel",
    "RetNetForImageClassification",
    "RetNetForMaskedImageModeling",
    "RetNetVisionConfig",
    "RetNetVideoModel",
    "RetNetForVideoClassification",
    "RetNetForVideoPreTraining",
    "RetNetVideoConfig",
]
