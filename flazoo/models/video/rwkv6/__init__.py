# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_rwkv6 import RWKV6VideoConfig
from .modeling_rwkv6 import(RWKV6VideoModel, RWKV6ForVideoClassification, RWKV6ForVideoPreTraining)

AutoConfig.register(RWKV6VideoConfig.model_type, RWKV6VideoConfig)
AutoModel.register(RWKV6VideoConfig, RWKV6VideoModel)
AutoModelForVideoClassification.register(RWKV6VideoConfig, RWKV6ForVideoClassification)
AutoModelForPreTraining.register(RWKV6VideoConfig, RWKV6ForVideoPreTraining)

__all__ = ['RWKV6VideoConfig', 'RWKV6VideoModel', 'RWKV6ForVideoClassification', 'RWKV6ForVideoPreTraining']