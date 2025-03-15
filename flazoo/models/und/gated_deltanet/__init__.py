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

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_gated_deltanet import GatedDeltaNetVideoConfig
from .modeling_gated_deltanet import(GatedDeltaNetVideoModel, GatedDeltaNetForVideoClassification, GatedDeltaNetForVideoPreTraining)

AutoConfig.register(GatedDeltaNetVideoConfig.model_type, GatedDeltaNetVideoConfig)
AutoModel.register(GatedDeltaNetVideoConfig, GatedDeltaNetVideoModel)
AutoModelForVideoClassification.register(GatedDeltaNetVideoConfig, GatedDeltaNetForVideoClassification)
AutoModelForPreTraining.register(GatedDeltaNetVideoConfig, GatedDeltaNetForVideoPreTraining)

__all__ = ['GatedDeltaNetVisionModel', 'GatedDeltaNetForImageClassification', 'GatedDeltaNetForMaskedImageModeling', 'GatedDeltaNetVisionConfig', 'GatedDeltaNetVideoModel', 'GatedDeltaNetForVideoClassification', 'GatedDeltaNetForVideoPreTraining', 'GatedDeltaNetVideoConfig']
