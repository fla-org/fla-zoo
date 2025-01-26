# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_rwkv6 import RWKV6VisionConfig
from .modeling_rwkv6 import RWKV6VisionModel, RWKV6ForImageClassification, RWKV6ForMaskedImageModeling

AutoConfig.register(RWKV6VisionConfig.model_type, RWKV6VisionConfig)
AutoModel.register(RWKV6VisionConfig, RWKV6VisionModel)
AutoModelForImageClassification.register(RWKV6VisionConfig, RWKV6ForImageClassification)
AutoModelForMaskedImageModeling.register(RWKV6VisionConfig, RWKV6ForMaskedImageModeling)


__all__ = ['RWKV6VisionModel', 'RWKV6ForImageClassification', 'RWKV6ForMaskedImageModeling', 'RWKV6VisionConfig']
