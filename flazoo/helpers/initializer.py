import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
import copy
import logging

logger = logging.getLogger(__name__)

def initialize_from_pretrained(
    model_a: nn.Module,
    model_b: nn.Module,
    param_mapping: Dict[str, str],
    backbone_attr: Optional[List[str]] = None
) -> nn.Module:
    """
    Initialize parameters of model_a using parameters from pretrained model_b based on mapping.

    Args:
        model_a: Target model to be initialized
        model_b: Source pretrained model
        param_mapping: Dictionary mapping parameter names from model_a to model_b
        backbone_attr: List of attribute names to look for backbone (default: ['blocks', 'layers'])

    Returns:
        Initialized model_a
    """
    if backbone_attr is None:
        backbone_attr = ['blocks', 'layers']

    # Find backbone in both models
    backbone_a = _find_backbone(model_a, backbone_attr)
    backbone_b = _find_backbone(model_b, backbone_attr)

    if backbone_a is None or backbone_b is None:
        raise ValueError(f"Could not find backbone in models. Looked for attributes: {backbone_attr}")

    # Get number of layers in each model
    num_layers_a = len(backbone_a)
    num_layers_b = len(backbone_b)

    logger.info(f"Found backbones with {num_layers_a} and {num_layers_b} layers")

    # Check if we have enough layers in model_b
    if num_layers_a > num_layers_b:
        raise ValueError(f"Model A has more layers ({num_layers_a}) than Model B ({num_layers_b})")

    # Initialize parameters layer by layer
    for layer_idx in range(num_layers_a):
        layer_a = backbone_a[layer_idx]
        layer_b = backbone_b[layer_idx]

        # Apply parameter mapping
        for param_a, param_b in param_mapping.items():
            _initialize_parameter_in_a_layer(layer_a, layer_b, param_a, param_b)

    return model_a

def _find_backbone(model: nn.Module, backbone_attr: List[str]) -> Optional[nn.ModuleList]:
    """
    Find the backbone component of a model.

    Args:
        model: Model to search
        backbone_attr: List of attribute names to look for

    Returns:
        ModuleList containing the model layers or None if not found
    """
    # First check if model has encoder with blocks
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'blocks'):
        return model.encoder.blocks

    # Then check direct attributes
    for attr in backbone_attr:
        if hasattr(model, attr):
            return getattr(model, attr)

    return None

def _initialize_parameter_in_a_layer(
    layer_a: nn.Module,
    layer_b: nn.Module,
    param_a_name: str,
    param_b_name: str
) -> None:
    """
    Initialize a parameter in layer_a using the corresponding parameter from layer_b.

    Args:
        layer_a: Target layer
        layer_b: Source layer
        param_a_name: Parameter name in layer_a
        param_b_name: Parameter name in layer_b
    """
    # Check if parameters exist
    if not hasattr(layer_a, param_a_name):
        raise ValueError(f"Parameter {param_a_name} not found in target layer")

    if not hasattr(layer_b, param_b_name):
        raise ValueError(f"Parameter {param_b_name} not found in source layer")

    # Get parameters
    param_a = getattr(layer_a, param_a_name)
    param_b = getattr(layer_b, param_b_name)

    # Simplified for common module types like nn.Linear and nn.Conv2d
    if isinstance(param_a, nn.Module) and isinstance(param_b, nn.Module):
        # For common modules with weight and bias
        if hasattr(param_a, 'weight') and hasattr(param_b, 'weight'):
            # Check and copy weight
            if param_a.weight.shape != param_b.weight.shape:
                raise ValueError(f"Weight shape mismatch: {param_a.weight.shape} vs {param_b.weight.shape}")
            with torch.no_grad():
                param_a.weight.copy_(param_b.weight.data.clone())

            # Check and copy bias if present
            if hasattr(param_a, 'bias') and hasattr(param_b, 'bias') and param_a.bias is not None and param_b.bias is not None:
                if param_a.bias.shape != param_b.bias.shape:
                    raise ValueError(f"Bias shape mismatch: {param_a.bias.shape} vs {param_b.bias.shape}")
                with torch.no_grad():
                    param_a.bias.copy_(param_b.bias.data.clone())
        else:
            raise ValueError(f"Modules without weight attribute are not supported")

    elif isinstance(param_a, nn.Parameter) and isinstance(param_b, nn.Parameter):
        # Direct parameter to parameter copy
        if param_a.shape != param_b.shape:
            raise ValueError(f"Parameter shape mismatch: {param_a.shape} vs {param_b.shape}")
        with torch.no_grad():
            param_a.copy_(param_b.data.clone())

    elif isinstance(param_a, nn.Module) and isinstance(param_b, nn.Parameter):
        # Module <- Parameter (copy to weight)
        if hasattr(param_a, 'weight'):
            if param_a.weight.shape != param_b.shape:
                raise ValueError(f"Shape mismatch: {param_a.weight.shape} vs {param_b.shape}")
            with torch.no_grad():
                param_a.weight.copy_(param_b.data.clone())
        else:
            raise ValueError(f"Module without weight attribute not supported")

    elif isinstance(param_a, nn.Parameter) and isinstance(param_b, nn.Module):
        # Parameter <- Module (copy from weight)
        if hasattr(param_b, 'weight'):
            if param_a.shape != param_b.weight.shape:
                raise ValueError(f"Shape mismatch: {param_a.shape} vs {param_b.weight.shape}")
            with torch.no_grad():
                param_a.copy_(param_b.weight.data.clone())
        else:
            raise ValueError(f"Module without weight attribute not supported")

    else:
        raise ValueError(f"Incompatible types: {type(param_a)} and {type(param_b)}")

def initialize_attention_params(
    model_a: nn.Module,
    model_b: nn.Module,
    q_mapping: str = "q_proj",
    k_mapping: str = "k_proj",
    v_mapping: str = "v_proj",
    o_mapping: str = "o_proj",
    backbone_attr: Optional[List[str]] = None
) -> nn.Module:
    """
    Initialize attention parameters of model_a using parameters from pretrained model_b.

    Args:
        model_a: Target model to be initialized
        model_b: Source pretrained model
        q_mapping: Name of query projection in both models
        k_mapping: Name of key projection in both models
        v_mapping: Name of value projection in both models
        o_mapping: Name of output projection in both models
        backbone_attr: List of attribute names to look for backbone

    Returns:
        Initialized model_a
    """
    # Create mapping dictionary for attention parameters
    mapping = {
        f"attn.{q_mapping}": f"attn.{q_mapping}",
        f"attn.{k_mapping}": f"attn.{k_mapping}",
        f"attn.{v_mapping}": f"attn.{v_mapping}",
        f"attn.{o_mapping}": f"attn.{o_mapping}"
    }

    return initialize_from_pretrained(model_a, model_b, mapping, backbone_attr)

def initialize_mlp_params(
    model_a: nn.Module,
    model_b: nn.Module,
    mlp_mapping: Dict[str, str],
    backbone_attr: Optional[List[str]] = None
) -> nn.Module:
    """
    Initialize MLP parameters of model_a using parameters from pretrained model_b.

    Args:
        model_a: Target model to be initialized
        model_b: Source pretrained model
        mlp_mapping: Dictionary mapping MLP parameter names from model_a to model_b
        backbone_attr: List of attribute names to look for backbone

    Returns:
        Initialized model_a
    """
    # Create mapping dictionary with mlp prefix
    mapping = {f"mlp.{k}": f"mlp.{v}" for k, v in mlp_mapping.items()}

    return initialize_from_pretrained(model_a, model_b, mapping, backbone_attr)

def initialize_custom_mapping(
    model_a: nn.Module,
    model_b: nn.Module,
    param_mapping: Dict[str, str],
    backbone_attr: Optional[List[str]] = None
) -> nn.Module:
    """
    Initialize parameters of model_a using parameters from pretrained model_b with custom mapping.

    Args:
        model_a: Target model to be initialized
        model_b: Source pretrained model
        param_mapping: Dictionary mapping parameter names from model_a to model_b
        backbone_attr: List of attribute names to look for backbone

    Returns:
        Initialized model_a
    """
    return initialize_from_pretrained(model_a, model_b, param_mapping, backbone_attr)