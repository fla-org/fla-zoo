# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_gla import GLAVisionConfig
from .modeling_gla import GLAVisionModel, GLAForImageClassification, GLAForMaskedImageModeling

AutoConfig.register(GLAVisionConfig.model_type, GLAVisionConfig)
AutoModelForImageClassification.register(GLAVisionConfig, GLAForImageClassification)
AutoModelForMaskedImageModeling.register(GLAVisionConfig, GLAForMaskedImageModeling)
AutoModel.register(GLAVisionConfig, GLAVisionModel)


__all__ = ['GLAVisionModel', 'GLAForImageClassification', 'GLAForMaskedImageModeling', 'GLAVisionConfig']
