import os
import sys
import torch
from transformers import AutoImageProcessor, AutoModel

from flazoo.models import DeltaNetVisionModel, DeltaNetVisionConfig
from flazoo.helpers.initializer import (
    initialize_custom_mapping
)


def create_model(num_layers, hidden_size):
    """Create a model with the specified configuration."""
    config = {
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "image_size": 224,
        "patch_size": 16,
    }
    config = DeltaNetVisionConfig(**config)
    return DeltaNetVisionModel(config)

def main():    

    model_a = create_model(12, 768)
    model_b = AutoModel.from_pretrained('facebook/dinov2-base')
    
    print("Initializing with custom parameter mapping...")
    # Example of custom parameter mapping
    custom_mapping = {
        "attn.q_proj": "attention.attention.query",
        "attn.k_proj": "attention.attention.key",
        "attn.v_proj": "attention.attention.value",
        "attn.o_proj": "attention.output.dense",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2"
    }
    initialize_custom_mapping(
        model_a=model_a,
        model_b=model_b,
        param_mapping=custom_mapping
    )
    # some random checking
    for i in range(12):
        tensor1 = model_b.encoder.layer[i].attention.attention.key.weight 
        tensor2 = model_a.encoder.blocks[i].attn.k_proj.weight  
        # check whether the two tensors are equal
        assert torch.equal(tensor1, tensor2)
        print(f"Layer {i} initialized successfully!")
    print("Initialization complete!")

if __name__ == "__main__":
    main()
