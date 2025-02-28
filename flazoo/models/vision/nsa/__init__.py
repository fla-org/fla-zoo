# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_nsa import NSAVisionConfig
from .modeling_nsa import(NSAVisionModel,
                                            NSAForImageClassification,
                                            NSAForMaskedImageModeling)

AutoConfig.register(NSAVisionConfig.model_type, NSAVisionConfig)
AutoModel.register(NSAVisionConfig, NSAVisionModel)
AutoModelForImageClassification.register(NSAVisionConfig, NSAForImageClassification)
AutoModelForMaskedImageModeling.register(NSAVisionConfig, NSAForMaskedImageModeling)

__all__ = ['NSAVisionModel', 'NSAForImageClassification', 'NSAForMaskedImageModeling', 'NSAVisionConfig']
