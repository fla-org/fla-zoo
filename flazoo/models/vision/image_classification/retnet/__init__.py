# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_retnet import RetNetVisionConfig
from .modeling_retnet import RetNetVisionModel, RetNetForImageClassification, RetNetForMaskedImageModeling

AutoConfig.register(RetNetVisionConfig.model_type, RetNetVisionConfig)
AutoModel.register(RetNetVisionConfig, RetNetVisionModel)
AutoModelForImageClassification.register(RetNetVisionConfig, RetNetForImageClassification)
AutoModelForMaskedImageModeling.register(RetNetVisionConfig, RetNetForMaskedImageModeling)


__all__ = ['RetNetVisionModel', 'RetNetForImageClassification', 'RetNetForMaskedImageModeling', 'RetNetVisionConfig']
