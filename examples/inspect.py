import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from flazoo.helpers.informer import log_model_parameters_flat, log_model_parameters

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

log_model_parameters(model)
log_model_parameters_flat(model)
