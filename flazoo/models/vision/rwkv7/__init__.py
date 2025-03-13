# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_rwkv7 import RWKV7VisionConfig
from .modeling_rwkv7 import RWKV7VisionModel, RWKV7ForImageClassification, RWKV7ForMaskedImageModeling

AutoConfig.register(RWKV7VisionConfig.model_type, RWKV7VisionConfig)
AutoModel.register(RWKV7VisionConfig, RWKV7VisionModel)
AutoModelForImageClassification.register(RWKV7VisionConfig, RWKV7ForImageClassification)
AutoModelForMaskedImageModeling.register(RWKV7VisionConfig, RWKV7ForMaskedImageModeling)


__all__ = ['RWKV7VisionModel', 'RWKV7ForImageClassification', 'RWKV7ForMaskedImageModeling', 'RWKV7VisionConfig']
