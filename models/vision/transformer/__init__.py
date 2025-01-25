# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_transformer import TransformerVisionConfig
from .modeling_transformer import (
    TransformerVisionModel, TransformerForImageClassification, TransformerForMaskedImageModeling)

AutoConfig.register(TransformerVisionConfig.model_type, TransformerVisionConfig)
AutoModelForImageClassification.register(TransformerVisionConfig, TransformerForImageClassification)
AutoModelForMaskedImageModeling.register(TransformerVisionConfig, TransformerForMaskedImageModeling)
AutoModel.register(TransformerVisionConfig, TransformerVisionModel)


__all__ = ['TransformerVisionModel', 'TransformerForImageClassification', 'TransformerForMaskedImageModeling', 'TransformerVisionConfig']
