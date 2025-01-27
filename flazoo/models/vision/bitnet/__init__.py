# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_bitnet import BitNetVisionConfig
from .modeling_bitnet import BitNetVisionModel, BitNetForImageClassification, BitNetForMaskedImageModeling

AutoConfig.register(BitNetVisionConfig.model_type, BitNetVisionConfig)
AutoModelForImageClassification.register(BitNetVisionConfig, BitNetForImageClassification)
AutoModelForMaskedImageModeling.register(BitNetVisionConfig, BitNetForMaskedImageModeling)
AutoModel.register(BitNetVisionConfig, BitNetVisionModel)


__all__ = ['BitNetVisionConfig', 'BitNetForImageClassification', 'BitNetForMaskedImageModeling', 'BitNetVisionModel']
