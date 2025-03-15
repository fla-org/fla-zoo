# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from .configuration_delta_net import DeltaNetVisionConfig
from .modeling_delta_net import(DeltaNetVisionModel,
                                            DeltaNetForImageClassification,
                                            DeltaNetForMaskedImageModeling)

AutoConfig.register(DeltaNetVisionConfig.model_type, DeltaNetVisionConfig)
AutoModel.register(DeltaNetVisionConfig, DeltaNetVisionModel)
AutoModelForImageClassification.register(DeltaNetVisionConfig, DeltaNetForImageClassification)
AutoModelForMaskedImageModeling.register(DeltaNetVisionConfig, DeltaNetForMaskedImageModeling)

from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification, AutoModelForPreTraining

from .configuration_delta_net import DeltaNetVideoConfig
from .modeling_delta_net import(DeltaNetVideoModel, DeltaNetForVideoClassification, DeltaNetForVideoPreTraining)

AutoConfig.register(DeltaNetVideoConfig.model_type, DeltaNetVideoConfig)
AutoModel.register(DeltaNetVideoConfig, DeltaNetVideoModel)
AutoModelForVideoClassification.register(DeltaNetVideoConfig, DeltaNetForVideoClassification)
AutoModelForPreTraining.register(DeltaNetVideoConfig, DeltaNetForVideoPreTraining)

__all__ = ['DeltaNetVisionModel', 'DeltaNetForImageClassification', 'DeltaNetForMaskedImageModeling', 'DeltaNetVisionConfig', 'DeltaNetVideoModel', 'DeltaNetForVideoClassification', 'DeltaNetForVideoPreTraining', 'DeltaNetVideoConfig']
