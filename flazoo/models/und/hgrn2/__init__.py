# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_hgrn2 import HGRN2VisionConfig
from .modeling_hgrn2 import HGRN2VisionModel, HGRN2ForImageClassification, HGRN2ForMaskedImageModeling

AutoConfig.register(HGRN2VisionConfig.model_type, HGRN2VisionConfig)
AutoModelForImageClassification.register(HGRN2VisionConfig, HGRN2ForImageClassification)
AutoModelForMaskedImageModeling.register(HGRN2VisionConfig, HGRN2ForMaskedImageModeling)
AutoModel.register(HGRN2VisionConfig, HGRN2VisionModel)

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_hgrn2 import HGRN2VideoConfig
from .modeling_hgrn2 import(HGRN2VideoModel, HGRN2ForVideoClassification, HGRN2ForVideoPreTraining)

AutoConfig.register(HGRN2VideoConfig.model_type, HGRN2VideoConfig)
AutoModel.register(HGRN2VideoConfig, HGRN2VideoModel)
AutoModelForVideoClassification.register(HGRN2VideoConfig, HGRN2ForVideoClassification)
AutoModelForPreTraining.register(HGRN2VideoConfig, HGRN2ForVideoPreTraining)

__all__ = ['HGRN2VisionModel', 'HGRN2ForImageClassification', 'HGRN2ForMaskedImageModeling', 'HGRN2VisionConfig', 'HGRN2VideoModel', 'HGRN2ForVideoClassification', 'HGRN2ForVideoPreTraining', 'HGRN2VideoConfig']
