# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_gated_deltanet import \
    GatedDeltaNetVisionConfig
from .modeling_gated_deltanet import (
    GatedDeltaNetVisionModel, GatedDeltaNetForImageClassification, GatedDeltaNetForMaskedImageModeling)

AutoConfig.register(GatedDeltaNetVisionConfig.model_type, GatedDeltaNetVisionConfig)
AutoModelForImageClassification.register(GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification)
AutoModelForMaskedImageModeling.register(GatedDeltaNetVisionConfig, GatedDeltaNetForMaskedImageModeling)
AutoModel.register(GatedDeltaNetVisionConfig, GatedDeltaNetVisionModel)

__all__ = ['GatedDeltaNetVisionModel', 'GatedDeltaNetForImageClassification', 'GatedDeltaNetForMaskedImageModeling', 'GatedDeltaNetVisionConfig']
