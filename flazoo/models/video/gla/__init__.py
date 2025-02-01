# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_gla import GLAVideoConfig
from .modeling_gla import(GLAVideoModel, GLAForVideoClassification, GLAForVideoPreTraining)


AutoConfig.register(GLAVideoConfig.model_type, GLAVideoConfig)
AutoModel.register(GLAVideoConfig, GLAVideoModel)
AutoModelForVideoClassification.register(GLAVideoConfig, GLAForVideoClassification)
AutoModelForPreTraining.register(GLAVideoConfig, GLAForVideoPreTraining)

__all__ = ['GLAVideoConfig', 'GLAVideoModel', 'GLAForVideoClassification', 'GLAForVideoPreTraining']