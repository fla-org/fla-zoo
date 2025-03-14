# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_rwkv7 import RWKV7VideoConfig
from .modeling_rwkv7 import(RWKV7VideoModel, RWKV7ForVideoClassification, RWKV7ForVideoPreTraining)

AutoConfig.register(RWKV7VideoConfig.model_type, RWKV7VideoConfig)
AutoModel.register(RWKV7VideoConfig, RWKV7VideoModel)
AutoModelForVideoClassification.register(RWKV7VideoConfig, RWKV7ForVideoClassification)
AutoModelForPreTraining.register(RWKV7VideoConfig, RWKV7ForVideoPreTraining)

__all__ = ['RWKV7VideoConfig', 'RWKV7VideoModel', 'RWKV7ForVideoClassification', 'RWKV7ForVideoPreTraining']