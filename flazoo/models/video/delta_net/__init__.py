# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_delta_net import DeltaNetVideoConfig
from .modeling_delta_net import(DeltaNetVideoModel, DeltaNetForVideoClassification, DeltaNetForVideoPreTraining)

AutoConfig.register(DeltaNetVideoConfig.model_type, DeltaNetVideoConfig)
AutoModel.register(DeltaNetVideoConfig, DeltaNetVideoModel)
AutoModelForVideoClassification.register(DeltaNetVideoConfig, DeltaNetForVideoClassification)
AutoModelForPreTraining.register(DeltaNetVideoConfig, DeltaNetForVideoPreTraining)

__all__ = ['DeltaNetVideoConfig', 'DeltaNetVideoModel', 'DeltaNetForVideoClassification', 'DeltaNetForVideoPreTraining']