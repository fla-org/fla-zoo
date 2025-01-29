# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_retnet import RetNetVideoConfig
from .modeling_retnet import(RetNetVideoModel, RetNetForVideoClassification, RetNetForVideoPreTraining)

AutoConfig.register(RetNetVideoConfig.model_type, RetNetVideoConfig)
AutoModel.register(RetNetVideoConfig, RetNetVideoModel)
AutoModelForVideoClassification.register(RetNetVideoConfig, RetNetForVideoClassification)
AutoModelForPreTraining.register(RetNetVideoConfig, RetNetForVideoPreTraining)

__all__ = ['RetNetVideoConfig', 'RetNetVideoModel', 'RetNetForVideoClassification', 'RetNetForVideoPreTraining']