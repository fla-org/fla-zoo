# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_mesa_net import MesaNetVisionConfig
from .modeling_mesa_net import (
    MesaNetVisionModel,
    MesaNetForImageClassification,
    MesaNetForMaskedImageModeling,
)

AutoConfig.register(MesaNetVisionConfig.model_type, MesaNetVisionConfig)
AutoModel.register(MesaNetVisionConfig, MesaNetVisionModel)
AutoModelForImageClassification.register(
    MesaNetVisionConfig, MesaNetForImageClassification
)
AutoModelForMaskedImageModeling.register(
    MesaNetVisionConfig, MesaNetForMaskedImageModeling
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_mesa_net import MesaNetVideoConfig
from .modeling_mesa_net import (
    MesaNetVideoModel,
    MesaNetForVideoClassification,
    MesaNetForVideoPreTraining,
)

AutoConfig.register(MesaNetVideoConfig.model_type, MesaNetVideoConfig)
AutoModel.register(MesaNetVideoConfig, MesaNetVideoModel)
AutoModelForVideoClassification.register(
    MesaNetVideoConfig, MesaNetForVideoClassification
)
AutoModelForPreTraining.register(MesaNetVideoConfig, MesaNetForVideoPreTraining)

__all__ = [
    "MesaNetVisionModel",
    "MesaNetForImageClassification",
    "MesaNetForMaskedImageModeling",
    "MesaNetVisionConfig",
    "MesaNetVideoModel",
    "MesaNetForVideoClassification",
    "MesaNetForVideoPreTraining",
    "MesaNetVideoConfig",
]
