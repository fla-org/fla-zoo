# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_gated_deltanet import GatedDeltaNetVideoConfig
from .modeling_gated_deltanet import(GatedDeltaNetVideoModel, GatedDeltaNetForVideoClassification, GatedDeltaNetForVideoPreTraining)

AutoConfig.register(GatedDeltaNetVideoConfig.model_type, GatedDeltaNetVideoConfig)
AutoModel.register(GatedDeltaNetVideoConfig, GatedDeltaNetVideoModel)
AutoModelForVideoClassification.register(GatedDeltaNetVideoConfig, GatedDeltaNetForVideoClassification)
AutoModelForPreTraining.register(GatedDeltaNetVideoConfig, GatedDeltaNetForVideoPreTraining)

__all__ = ['GatedDeltaNetVideoConfig', 'GatedDeltaNetVideoModel', 'GatedDeltaNetForVideoClassification', 'GatedDeltaNetForVideoPreTraining']