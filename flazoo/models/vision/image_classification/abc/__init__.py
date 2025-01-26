# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_abc import ABCVisionConfig
from .modeling_abc import ABCVisionModel, ABCForImageClassification, ABCForMaskedImageModeling

AutoConfig.register(ABCVisionConfig.model_type, ABCVisionConfig)
AutoModelForImageClassification.register(ABCVisionConfig, ABCForImageClassification)
AutoModelForMaskedImageModeling.register(ABCVisionConfig, ABCForMaskedImageModeling)
AutoModel.register(ABCVisionConfig, ABCVisionModel)


__all__ = ['ABCVisionModel', 'ABCForImageClassification', 'ABCForMaskedImageModeling', 'ABCVisionConfig']
