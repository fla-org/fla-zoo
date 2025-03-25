# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_moba import MoBAVisionConfig
from .modeling_moba import(MoBAVisionModel,
                                            MoBAForImageClassification,
                                            MoBAForMaskedImageModeling)

AutoConfig.register(MoBAVisionConfig.model_type, MoBAVisionConfig)
AutoModel.register(MoBAVisionConfig, MoBAVisionModel)
AutoModelForImageClassification.register(MoBAVisionConfig, MoBAForImageClassification)
AutoModelForMaskedImageModeling.register(MoBAVisionConfig, MoBAForMaskedImageModeling)

__all__ = ['MoBAVisionModel', 'MoBAForImageClassification', 'MoBAForMaskedImageModeling', 'MoBAVisionConfig']
