# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_xattn import XAttentionVisionConfig
from .modeling_xattn import(XAttentionVisionModel,
                                            XAttentionForImageClassification,
                                            XAttentionForMaskedImageModeling)

AutoConfig.register(XAttentionVisionConfig.model_type, XAttentionVisionConfig)
AutoModel.register(XAttentionVisionConfig, XAttentionVisionModel)
AutoModelForImageClassification.register(XAttentionVisionConfig, XAttentionForImageClassification)
AutoModelForMaskedImageModeling.register(XAttentionVisionConfig, XAttentionForMaskedImageModeling)

__all__ = ['XAttentionVisionModel', 'XAttentionForImageClassification', 'XAttentionForMaskedImageModeling', 'XAttentionVisionConfig']
