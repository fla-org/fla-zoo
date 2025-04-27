import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from flazoo.helpers.informer import log_model_parameters_flat, log_model_parameters

from transformers import Siglip2ForImageClassification
model = AutoModel.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224').vision_model

log_model_parameters(model)
log_model_parameters_flat(model)
