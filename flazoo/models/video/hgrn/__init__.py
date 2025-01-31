# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_hgrn import HGRNVideoConfig
from .modeling_hgrn import(HGRNVideoModel, HGRNForVideoClassification, HGRNForVideoPreTraining)

AutoConfig.register(HGRNVideoConfig.model_type, HGRNVideoConfig)
AutoModel.register(HGRNVideoConfig, HGRNVideoModel)
AutoModelForVideoClassification.register(HGRNVideoConfig, HGRNForVideoClassification)
AutoModelForPreTraining.register(HGRNVideoConfig, HGRNForVideoPreTraining)

__all__ = ['HGRNVideoConfig', 'HGRNVideoModel', 'HGRNForVideoClassification', 'HGRNForVideoPreTraining']