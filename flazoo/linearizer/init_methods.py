from transformers import AutoModel
from flazoo.helpers.initializer import (
    initialize_custom_mapping
)
import torch

"""
一些预制函数
"""

def init_from_dino2_base(
    fla_model,
    dino_model: str = 'facebook/dinov2-base',
    train_mlp: bool = False,
):
    """
    Initialize a FLA model from a DINO model. \n
    Note that dinov2-base use patch_size=14

    Args:
        fla_model: FLA models to be initialized
        dino_model: Name or path of the DINO model to load
        verify: Whether to verify the initialization

    Returns:
        Initialized FLA model
    """

    dino = AutoModel.from_pretrained(dino_model)
    
    # Define parameter mapping
    param_mapping = {
        "attn.q_proj": "attention.attention.query",
        "attn.k_proj": "attention.attention.key",
        "attn.v_proj": "attention.attention.value",
        "attn.o_proj": "attention.output.dense",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2"
    }

    # Initialize parameters
    initialize_custom_mapping(
        model_a=fla_model,
        model_b=dino,
        param_mapping=param_mapping
    )

    # Optionally freeze MLP layers

    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    return fla_model

def init_from_siglip2_base_p16_224(
    fla_model,
    siglip_model: str = 'google/siglip2-base-patch16-224',
    train_mlp: bool = False,
):
    """
    Initialize a FLA model from a SigLIP2 model.

    Args:
        fla_model: FLA models to be initialized
        siglip_model: Name or path of the SigLIP2 model to load
        train_mlp: Whether to train the MLP layers (default: False)

    Returns:
        Initialized FLA model
    """
    # Load SigLIP2 model and get vision component
    siglip = AutoModel.from_pretrained(siglip_model).vision_model
    
    # Define parameter mapping from FLA to SigLIP2
    param_mapping = {
        "attn.q_proj": "self_attn.q_proj",
        "attn.k_proj": "self_attn.k_proj",
        "attn.v_proj": "self_attn.v_proj",
        "attn.o_proj": "self_attn.out_proj",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2"
    }

    # Initialize parameters
    initialize_custom_mapping(
        model_a=fla_model,
        model_b=siglip,
        param_mapping=param_mapping
    )

    # Optionally freeze MLP layers
    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    return fla_model