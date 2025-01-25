# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_hgrn import HGRNVisionConfig
from .modeling_hgrn import HGRNVisionModel, HGRNForImageClassification, HGRNForMaskedImageModeling

AutoConfig.register(HGRNVisionConfig.model_type, HGRNVisionConfig)
AutoModelForImageClassification.register(HGRNVisionConfig, HGRNForImageClassification)
AutoModelForMaskedImageModeling.register(HGRNVisionConfig, HGRNForMaskedImageModeling)
AutoModel.register(HGRNVisionConfig, HGRNVisionModel)


__all__ = ['HGRNVisionModel', 'HGRNForImageClassification', 'HGRNForMaskedImageModeling', 'HGRNVisionConfig']
