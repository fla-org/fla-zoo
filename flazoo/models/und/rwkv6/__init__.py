# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_rwkv6 import RWKV6VisionConfig
from .modeling_rwkv6 import RWKV6VisionModel, RWKV6ForImageClassification, RWKV6ForMaskedImageModeling

AutoConfig.register(RWKV6VisionConfig.model_type, RWKV6VisionConfig)
AutoModel.register(RWKV6VisionConfig, RWKV6VisionModel)
AutoModelForImageClassification.register(RWKV6VisionConfig, RWKV6ForImageClassification)
AutoModelForMaskedImageModeling.register(RWKV6VisionConfig, RWKV6ForMaskedImageModeling)

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_rwkv6 import RWKV6VideoConfig
from .modeling_rwkv6 import(RWKV6VideoModel, RWKV6ForVideoClassification, RWKV6ForVideoPreTraining)

AutoConfig.register(RWKV6VideoConfig.model_type, RWKV6VideoConfig)
AutoModel.register(RWKV6VideoConfig, RWKV6VideoModel)
AutoModelForVideoClassification.register(RWKV6VideoConfig, RWKV6ForVideoClassification)
AutoModelForPreTraining.register(RWKV6VideoConfig, RWKV6ForVideoPreTraining)

__all__ = ['RWKV6VisionModel', 'RWKV6ForImageClassification', 'RWKV6ForMaskedImageModeling', 'RWKV6VisionConfig', 'RWKV6VideoModel', 'RWKV6ForVideoClassification', 'RWKV6ForVideoPreTraining', 'RWKV6VideoConfig']
