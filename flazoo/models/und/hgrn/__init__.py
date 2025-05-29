# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_hgrn import HGRNVisionConfig
from .modeling_hgrn import (
    HGRNVisionModel,
    HGRNForImageClassification,
    HGRNForMaskedImageModeling,
)

AutoConfig.register(HGRNVisionConfig.model_type, HGRNVisionConfig)
AutoModelForImageClassification.register(HGRNVisionConfig, HGRNForImageClassification)
AutoModelForMaskedImageModeling.register(HGRNVisionConfig, HGRNForMaskedImageModeling)
AutoModel.register(HGRNVisionConfig, HGRNVisionModel)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_hgrn import HGRNVideoConfig
from .modeling_hgrn import (
    HGRNVideoModel,
    HGRNForVideoClassification,
    HGRNForVideoPreTraining,
)

AutoConfig.register(HGRNVideoConfig.model_type, HGRNVideoConfig)
AutoModel.register(HGRNVideoConfig, HGRNVideoModel)
AutoModelForVideoClassification.register(HGRNVideoConfig, HGRNForVideoClassification)
AutoModelForPreTraining.register(HGRNVideoConfig, HGRNForVideoPreTraining)

__all__ = [
    "HGRNVisionModel",
    "HGRNForImageClassification",
    "HGRNForMaskedImageModeling",
    "HGRNVisionConfig",
    "HGRNVideoModel",
    "HGRNForVideoClassification",
    "HGRNForVideoPreTraining",
    "HGRNVideoConfig",
]
