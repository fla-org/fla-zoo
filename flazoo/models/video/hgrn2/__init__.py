# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_hgrn2 import HGRN2VideoConfig
from .modeling_hgrn2 import(HGRN2VideoModel, HGRN2ForVideoClassification, HGRN2ForVideoPreTraining)

AutoConfig.register(HGRN2VideoConfig.model_type, HGRN2VideoConfig)
AutoModel.register(HGRN2VideoConfig, HGRN2VideoModel)
AutoModelForVideoClassification.register(HGRN2VideoConfig, HGRN2ForVideoClassification)
AutoModelForPreTraining.register(HGRN2VideoConfig, HGRN2ForVideoPreTraining)

__all__ = ['HGRN2VideoConfig', 'HGRN2VideoModel', 'HGRN2ForVideoClassification', 'HGRN2ForVideoPreTraining']