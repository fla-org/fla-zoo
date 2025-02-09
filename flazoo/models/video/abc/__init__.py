# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_abc import ABCVideoConfig
from .modeling_abc import(ABCVideoModel, ABCForVideoClassification, ABCForVideoPreTraining)

AutoConfig.register(ABCVideoConfig.model_type, ABCVideoConfig)
AutoModel.register(ABCVideoConfig, ABCVideoModel)
AutoModelForVideoClassification.register(ABCVideoConfig, ABCForVideoClassification)
AutoModelForPreTraining.register(ABCVideoConfig, ABCForVideoPreTraining)

__all__ = ['ABCVideoConfig', 'ABCVideoModel', 'ABCForVideoClassification', 'ABCForVideoPreTraining']