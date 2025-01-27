# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_gsa import GSAVisionConfig
from .modeling_gsa import GSAVisionModel, GSAForImageClassification, GSAForMaskedImageModeling

AutoConfig.register(GSAVisionConfig.model_type, GSAVisionConfig)
AutoModelForImageClassification.register(GSAVisionConfig, GSAForImageClassification)
AutoModelForMaskedImageModeling.register(GSAVisionConfig, GSAForMaskedImageModeling)
AutoModel.register(GSAVisionConfig, GSAVisionModel)

__all__ = ['GSAVisionModel', 'GSAForImageClassification', 'GSAForMaskedImageModeling', 'GSAVisionConfig']
