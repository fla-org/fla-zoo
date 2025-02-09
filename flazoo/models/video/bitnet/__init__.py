# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_bitnet import BitNetVideoConfig
from .modeling_bitnet import(BitNetVideoModel, BitNetForVideoClassification, BitNetForVideoPreTraining)

AutoConfig.register(BitNetVideoConfig.model_type, BitNetVideoConfig)
AutoModel.register(BitNetVideoConfig, BitNetVideoModel)
AutoModelForVideoClassification.register(BitNetVideoConfig, BitNetForVideoClassification)
AutoModelForPreTraining.register(BitNetVideoConfig, BitNetForVideoPreTraining)

__all__ = ['BitNetVideoConfig', 'BitNetVideoModel', 'BitNetForVideoClassification', 'BitNetForVideoPreTraining']