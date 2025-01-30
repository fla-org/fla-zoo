# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_gsa import GSAVideoConfig
from .modeling_gsa import(GSAVideoModel, GSAForVideoClassification, GSAForVideoPreTraining)


AutoConfig.register(GSAVideoConfig.model_type, GSAVideoConfig)
AutoModel.register(GSAVideoConfig, GSAVideoModel)
AutoModelForVideoClassification.register(GSAVideoConfig, GSAForVideoClassification)
AutoModelForPreTraining.register(GSAVideoConfig, GSAForVideoPreTraining)

__all__ = ['GSAVideoConfig', 'GSAVideoModel', 'GSAForVideoClassification', 'GSAForVideoPreTraining']