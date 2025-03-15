# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_bitnet import BitNetVisionConfig
from .modeling_bitnet import BitNetVisionModel, BitNetForImageClassification, BitNetForMaskedImageModeling

AutoConfig.register(BitNetVisionConfig.model_type, BitNetVisionConfig)
AutoModelForImageClassification.register(BitNetVisionConfig, BitNetForImageClassification)
AutoModelForMaskedImageModeling.register(BitNetVisionConfig, BitNetForMaskedImageModeling)
AutoModel.register(BitNetVisionConfig, BitNetVisionModel)


from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_bitnet import BitNetVideoConfig
from .modeling_bitnet import(BitNetVideoModel, BitNetForVideoClassification, BitNetForVideoPreTraining)

AutoConfig.register(BitNetVideoConfig.model_type, BitNetVideoConfig)
AutoModel.register(BitNetVideoConfig, BitNetVideoModel)
AutoModelForVideoClassification.register(BitNetVideoConfig, BitNetForVideoClassification)
AutoModelForPreTraining.register(BitNetVideoConfig, BitNetForVideoPreTraining)

__all__ = ['BitNetVisionConfig', 'BitNetForImageClassification', 'BitNetForMaskedImageModeling', 'BitNetVisionModel', 'BitNetVideoConfig', 'BitNetVideoModel', 'BitNetForVideoClassification', 'BitNetForVideoPreTraining']
