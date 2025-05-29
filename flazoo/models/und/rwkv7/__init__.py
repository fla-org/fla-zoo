# -*- coding: utf-8 -*-

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
)

from .configuration_rwkv7 import RWKV7VisionConfig
from .modeling_rwkv7 import (
    RWKV7VisionModel,
    RWKV7ForImageClassification,
    RWKV7ForMaskedImageModeling,
)

AutoConfig.register(RWKV7VisionConfig.model_type, RWKV7VisionConfig)
AutoModel.register(RWKV7VisionConfig, RWKV7VisionModel)
AutoModelForImageClassification.register(RWKV7VisionConfig, RWKV7ForImageClassification)
AutoModelForMaskedImageModeling.register(RWKV7VisionConfig, RWKV7ForMaskedImageModeling)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVideoClassification,
    AutoModelForPreTraining,
)

from .configuration_rwkv7 import RWKV7VideoConfig
from .modeling_rwkv7 import (
    RWKV7VideoModel,
    RWKV7ForVideoClassification,
    RWKV7ForVideoPreTraining,
)

AutoConfig.register(RWKV7VideoConfig.model_type, RWKV7VideoConfig)
AutoModel.register(RWKV7VideoConfig, RWKV7VideoModel)
AutoModelForVideoClassification.register(RWKV7VideoConfig, RWKV7ForVideoClassification)
AutoModelForPreTraining.register(RWKV7VideoConfig, RWKV7ForVideoPreTraining)

__all__ = [
    "RWKV7VisionModel",
    "RWKV7ForImageClassification",
    "RWKV7ForMaskedImageModeling",
    "RWKV7VisionConfig",
    "RWKV7VideoModel",
    "RWKV7ForVideoClassification",
    "RWKV7ForVideoPreTraining",
    "RWKV7VideoConfig",
]
