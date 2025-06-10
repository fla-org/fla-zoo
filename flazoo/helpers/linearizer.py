from transformers import AutoModel, AutoConfig
from flazoo.helpers.initializer import initialize_custom_mapping
import torch
from torch import nn
import logging


"""
一些预制函数

This part is to init pre-defined FLA models using pretrained transformers.

This is for people who want to use FLA models. So the parts get inited are components in FLA models.

For example, SigLIP2 has some extra components like head.mlp.fc1.weight and head.mlp.fc12.weight \n

, which are not in FLA models. So they ignored. However, if you use linearized SigLIP2, you can use these extra components.

"""


def copy_matching_params(model_a, model_b, outlier_list=None, verbose=False):
    """
    Copy parameters from model_b to model_a where parameter names match,
    skipping parameters specified in outlier_list.

    Args:
        model_a: Target model whose parameters will be updated
        model_b: Source model whose parameters will be copied
        outlier_list: List of parameter names to skip (exact name matching)
        verbose: Whether to print detailed matching information

    Returns:
        tuple: (number of copied parameters, total parameters in model_a)
    """
    copied_params_count = 0
    total_params_count = 0

    outliers = set(outlier_list if outlier_list is not None else [])

    dict_a = dict(model_a.named_parameters())
    dict_b = dict(model_b.named_parameters())

    for name, param_a in model_a.named_parameters():
        total_params_count += param_a.numel()

        # Skip if parameter name is in outliers set (exact match)
        if name in outliers:
            if verbose:
                logging.info(f"Skipped parameter (in outlier list): {name}")
            continue

        # Check if parameter name exists in model_b
        if name in dict_b:
            logging.info(f"Matching parameter found: {name}")
            param_b = dict_b[name]

            # Check if parameter shapes match
            if param_a.shape == param_b.shape:
                # Copy parameter data
                param_a.data.copy_(param_b.data)
                copied_params_count += param_a.numel()
                if verbose:
                    logging.info(f"Copied parameter: {name}, shape: {param_a.shape}")
            else:
                if verbose:
                    logging.warning(
                        f"Parameter shape mismatch, skipped: {name}, "
                        f"model_a shape: {param_a.shape}, "
                        f"model_b shape: {param_b.shape}"
                    )

    # Print summary
    logging.info(
        f"Copied {copied_params_count}/{total_params_count} "
        f"parameters ({copied_params_count / total_params_count * 100:.2f}%)"
    )

    return copied_params_count, total_params_count


def init_video_und_from_hybrid_fla_vision_und(
    fla_model,
    another_fla_model,
    train_mlp: bool = True,
    init_embedding: bool = True,
    return_pretrained: bool = False,
):
    """
    Initialize a FLA model from another FLA model. \n
    This function is used to initialize a video FLA model from a hybrid FLA vision model. \n

    Args:
        fla_model: FLA models to be initialized
        another_fla_model: FLA models to load
        train_mlp: Whether to train the MLP layers (default: True)
        init_embedding: Whether to initialize the embedding layers (default: True)
        return_pretrained: Whether to return the pretrained model (default: False)

    Returns:
        Initialized FLA model
    """
    # Define parameter mapping
    outlier_list = [
        "embeddings.patch_embeddings.projection.weight",
        "embeddings.patch_embeddings.projection.bias",
    ]

    copy_matching_params(
        model_a=fla_model,
        model_b=another_fla_model,
        outlier_list=outlier_list,
        verbose=True,
    )

    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    if init_embedding:
        logging.info("Initializing embedding layers, make sure your shapes match.")
        fla_model.embeddings.patch_embeddings.projection.weight.data.copy_(
            another_fla_model.embeddings.patch_embeddings.projection.weight.data.unsqueeze(
                2
            )
        )
        fla_model.embeddings.patch_embeddings.projection.bias.data.copy_(
            another_fla_model.embeddings.patch_embeddings.projection.bias.data
        )
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.weight,
            another_fla_model.embeddings.patch_embeddings.projection.weight.unsqueeze(
                2
            ),
        )
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.bias,
            another_fla_model.embeddings.patch_embeddings.projection.bias,
        )

    # init layernorm weight and bias
    fla_model.layernorm.weight = nn.Parameter(
        another_fla_model.layernorm.weight.clone()
    )
    fla_model.layernorm.bias = nn.Parameter(another_fla_model.layernorm.bias.clone())
    fla_model.pooler.dense.weight = nn.Parameter(
        another_fla_model.pooler.dense.weight.clone()
    )
    fla_model.pooler.dense.bias = nn.Parameter(
        another_fla_model.pooler.dense.bias.clone()
    )

    if not return_pretrained:
        return fla_model
    else:
        return fla_model, another_fla_model


def init_video_und_from_pure_fla_vision_und(
    fla_model,
    another_fla_model,
    train_mlp: bool = True,
    init_embedding: bool = True,
    return_pretrained: bool = False,
):
    """
    Initialize a FLA model from another FLA model. \n
    This function is used to initialize a video FLA model from a pure FLA vision model. \n

    Args:
        fla_model: FLA models to be initialized
        another_fla_model: FLA models to load
        train_mlp: Whether to train the MLP layers (default: True)
        init_embedding: Whether to initialize the embedding layers (default: True)
        return_pretrained: Whether to return the pretrained model (default: False)

    Returns:
        Initialized FLA model
    """
    # Define parameter mapping
    param_mapping = {
        "attn.q_proj": "attn.q_proj",
        "attn.k_proj": "attn.k_proj",
        "attn.v_proj": "attn.v_proj",
        "attn.o_proj": "attn.o_proj",
        "attn.b_proj": "attn.b_proj",
        "attn.k_conv1d": "attn.k_conv1d",
        "attn.v_conv1d": "attn.v_conv1d",
        "attn.q_conv1d": "attn.q_conv1d",
        "attn.o_norm": "attn.o_norm",
        "ln_1": "ln_1",
        "ln_2": "ln_2",
        "channel_mixer.net.0": "channel_mixer.net.0",
        "channel_mixer.net.2": "channel_mixer.net.2",
    }

    initialize_custom_mapping(
        model_a=fla_model, model_b=another_fla_model, param_mapping=param_mapping
    )

    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    if init_embedding:
        logging.info("Initializing embedding layers, make sure your shapes match.")
        fla_model.embeddings.patch_embeddings.projection.weight.data.copy_(
            another_fla_model.embeddings.patch_embeddings.projection.weight.data.unsqueeze(
                2
            )
        )
        fla_model.embeddings.patch_embeddings.projection.bias.data.copy_(
            another_fla_model.embeddings.patch_embeddings.projection.bias.data
        )
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.weight,
            another_fla_model.embeddings.patch_embeddings.projection.weight.unsqueeze(
                2
            ),
        )
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.bias,
            another_fla_model.embeddings.patch_embeddings.projection.bias,
        )

    # init layernorm weight and bias
    fla_model.layernorm.weight = nn.Parameter(
        another_fla_model.layernorm.weight.clone()
    )
    fla_model.layernorm.bias = nn.Parameter(another_fla_model.layernorm.bias.clone())
    fla_model.pooler.dense.weight = nn.Parameter(
        another_fla_model.pooler.dense.weight.clone()
    )
    fla_model.pooler.dense.bias = nn.Parameter(
        another_fla_model.pooler.dense.bias.clone()
    )

    if not return_pretrained:
        return fla_model
    else:
        return fla_model, another_fla_model


def init_from_dino2_base_p14(
    fla_model,
    dino_model: str = "facebook/dinov2-base",
    train_mlp: bool = False,
    init_embedding: bool = True,
    return_pretrained: bool = False,
):
    """
    Initialize a FLA model from a DINO model. \n
    Note that dinov2-base use patch_size=14

    Args:
        fla_model: FLA models to be initialized
        dino_model: Name or path of the DINO model to load
        train_mlp: Whether to train the MLP layers (default: False)
        init_embedding: Whether to initialize the embedding layers (default: True)
        return_pretrained: Whether to return the pretrained model (default: False)

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
        "ln_1": "norm1",
        "ln_2": "norm2",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2",
    }

    # Initialize parameters
    initialize_custom_mapping(
        model_a=fla_model, model_b=dino, param_mapping=param_mapping
    )

    # Optionally freeze MLP layers

    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    if not return_pretrained:
        return fla_model
    else:
        return fla_model, dino


def init_from_dino2_small_p14(
    fla_model,
    dino_model: str = "facebook/dinov2-small",
    train_mlp: bool = False,
    init_embedding: bool = True,
    return_pretrained: bool = False,
):
    """
    Initialize a FLA model from a DINO model. \n
    Note that dinov2-small use patch_size=14

    Args:
        fla_model: FLA models to be initialized
        dino_model: Name or path of the DINO model to load
        train_mlp: Whether to train the MLP layers (default: False)
        init_embedding: Whether to initialize the embedding layers (default: True)
        return_pretrained: Whether to return the pretrained model (default: False)

    Returns:

    """
    dino = AutoModel.from_pretrained(dino_model)

    # Define parameter mapping
    param_mapping = {
        "attn.q_proj": "attention.attention.query",
        "attn.k_proj": "attention.attention.key",
        "attn.v_proj": "attention.attention.value",
        "attn.o_proj": "attention.output.dense",
        "ln_1": "norm1",
        "ln_2": "norm2",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2",
    }

    # Initialize parameters
    initialize_custom_mapping(
        model_a=fla_model, model_b=dino, param_mapping=param_mapping
    )

    # Optionally freeze MLP layers

    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    if not return_pretrained:
        return fla_model
    else:
        return fla_model, dino


def init_from_siglip2_base_p16_224(
    fla_model,
    custom_siglip_model=None,
    siglip_model: str = "google/siglip2-base-patch16-224",
    train_mlp: bool = False,
    init_embedding: bool = True,
    return_pretrained: bool = False,
    override_model: bool = False,
):
    """
    Initialize a FLA model from a SigLIP2 model.

    Args:
        fla_model: FLA models to be initialized
        siglip_model: Name or path of the SigLIP2 model to load
        train_mlp: Whether to train the MLP layers (default: False)
        init_embedding: Whether to initialize the embedding layers (default: True)
        init_head: Whether to initialize the head layers, useful for classification (default: True)
        return_pretrained: Whether to return the pretrained model (default: False)

    Returns:
        Initialized FLA model
    """
    # Load SigLIP2 model and get vision component
    if not override_model and custom_siglip_model is None:
        siglip = AutoModel.from_pretrained(siglip_model).vision_model
    else:
        if custom_siglip_model is not None:
            logging.info(
                "Overriding the model with custom SigLIP2 model, "
                "make sure you have the correct model structure."
            )
            siglip = custom_siglip_model
        else:
            # it's a string
            raise ValueError(
                "You must provide a custom SigLIP2 model or set override_model to True."
            )

    # Define parameter mapping from FLA to SigLIP2
    param_mapping = {
        "attn.q_proj": "self_attn.q_proj",
        "attn.k_proj": "self_attn.k_proj",
        "attn.v_proj": "self_attn.v_proj",
        "attn.o_proj": "self_attn.out_proj",
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2",
    }

    # Initialize parameters
    initialize_custom_mapping(
        model_a=fla_model, model_b=siglip, param_mapping=param_mapping
    )

    if init_embedding:
        logging.info("Initializing embedding layers, make sure your shapes match.")
        fla_model.embeddings.patch_embeddings.projection.weight.data.copy_(
            siglip.embeddings.patch_embedding.weight.data
        )
        fla_model.embeddings.patch_embeddings.projection.bias.data.copy_(
            siglip.embeddings.patch_embedding.bias.data
        )

        fla_model.embeddings.position_embeddings.data.copy_(
            siglip.embeddings.position_embedding.weight.data.unsqueeze(0)
        )

        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.weight,
            siglip.embeddings.patch_embedding.weight,
        )
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.bias,
            siglip.embeddings.patch_embedding.bias,
        )
        assert torch.equal(
            fla_model.embeddings.position_embeddings,
            siglip.embeddings.position_embedding.weight.unsqueeze(0),
        )

    # Optionally freeze MLP layers
    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    if not return_pretrained:
        return fla_model
    else:
        return fla_model, siglip


def init_from_clip_base_p16_224(
    fla_model,
    clip_model: str = "openai/clip-vit-base-patch16",
    train_mlp: bool = False,
    init_embedding: bool = True,
    return_pretrained: bool = False,
):
    """
    Initialize a FLA model from a CLIP model.

    Args:
        fla_model: FLA models to be initialized
        clip_model: Name or path of the clip model to load
        train_mlp: Whether to train the MLP layers (default: False)
        init_embedding: Whether to initialize the embedding layers (default: True)
        return_pretrained: Whether to return the pretrained model (default: False)

    Returns:
        Initialized FLA model
    """

    clip = AutoModel.from_pretrained(clip_model).vision_model

    param_mapping = {
        "attn.q_proj": "self_attn.q_proj",
        "attn.k_proj": "self_attn.k_proj",
        "attn.v_proj": "self_attn.v_proj",
        "attn.o_proj": "self_attn.out_proj",
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "channel_mixer.net.0": "mlp.fc1",
        "channel_mixer.net.2": "mlp.fc2",
    }

    initialize_custom_mapping(
        model_a=fla_model, model_b=clip, param_mapping=param_mapping
    )

    if init_embedding:
        # Copy patch embedding weights
        fla_model.embeddings.patch_embeddings.projection.weight.data.copy_(
            clip.embeddings.patch_embedding.weight.data
        )
        fla_model.embeddings.patch_embeddings.projection.bias.data.copy_(
            clip.embeddings.patch_embedding.bias.data
        )

        # Copy position embeddings, skipping the class token position (first token)
        # CLIP shape: (197, 768) -> (196, 768) -> (1, 196, 768)
        position_embeddings = clip.embeddings.position_embedding.weight.data[
            1:
        ].unsqueeze(0)
        fla_model.embeddings.position_embeddings.data.copy_(position_embeddings)

        # Verify the copying was successful
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.weight,
            clip.embeddings.patch_embedding.weight,
        )
        assert torch.equal(
            fla_model.embeddings.patch_embeddings.projection.bias,
            clip.embeddings.patch_embedding.bias,
        )
        assert torch.equal(
            fla_model.embeddings.position_embeddings,
            clip.embeddings.position_embedding.weight[1:].unsqueeze(0),
        )

    # Optionally freeze MLP layers
    if not train_mlp:
        for n, p in fla_model.named_parameters():
            if "channel_mixer" in n:
                p.requires_grad_(False)

    if not return_pretrained:
        return fla_model
    else:
        return fla_model, clip
