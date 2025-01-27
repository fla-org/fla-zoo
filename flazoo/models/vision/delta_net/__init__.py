# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_delta_net import DeltaNetVisionConfig
from .modeling_delta_net import(DeltaNetVisionModel,
                                            DeltaNetForImageClassification,
                                            DeltaNetForMaskedImageModeling)

AutoConfig.register(DeltaNetVisionConfig.model_type, DeltaNetVisionConfig)
AutoModel.register(DeltaNetVisionConfig, DeltaNetVisionModel)
AutoModelForImageClassification.register(DeltaNetVisionConfig, DeltaNetForImageClassification)
AutoModelForMaskedImageModeling.register(DeltaNetVisionConfig, DeltaNetForMaskedImageModeling)

__all__ = ['DeltaNetVisionModel', 'DeltaNetForImageClassification', 'DeltaNetForMaskedImageModeling', 'DeltaNetVisionConfig']
