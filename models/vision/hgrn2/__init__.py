# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_hgrn2 import HGRN2VisionConfig
from .modeling_hgrn2 import HGRN2VisionModel, HGRN2ForImageClassification, HGRN2ForMaskedImageModeling

AutoConfig.register(HGRN2VisionConfig.model_type, HGRN2VisionConfig)
AutoModelForImageClassification.register(HGRN2VisionConfig, HGRN2ForImageClassification)
AutoModelForMaskedImageModeling.register(HGRN2VisionConfig, HGRN2ForMaskedImageModeling)
AutoModel.register(HGRN2VisionConfig, HGRN2VisionModel)


__all__ = ['HGRN2VisionModel', 'HGRN2ForImageClassification', 'HGRN2ForMaskedImageModeling', 'HGRN2VisionConfig']
