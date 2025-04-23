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
        backbone_attr = ['blocks', 'layers', 'layer', 'block']

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

    # Recursively search for backbone attributes
    def search_backbone(module: nn.Module, depth: int = 0, max_depth: int = 5) -> Optional[nn.ModuleList]:
        if depth > max_depth:
            return None
            
        # Check direct attributes first
        for attr in backbone_attr:
            if hasattr(module, attr):
                backbone = getattr(module, attr)
                if isinstance(backbone, nn.ModuleList):
                    return backbone

        # Then recursively check all child modules
        for name, child in module.named_children():
            backbone = search_backbone(child, depth + 1, max_depth)
            if backbone is not None:
                return backbone
                
        return None

    # Start recursive search
    backbone = search_backbone(model)
    if backbone is not None:
        return backbone

    return None

def _get_module_by_path(module: nn.Module, path: str) -> nn.Module:
    """Get nested module using dot notation."""
    current = module
    for part in path.split('.'):
        current = getattr(current, part)
    return current

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

    try:
        param_a = _get_module_by_path(layer_a, param_a_name)
    except AttributeError:
        logger.warning(f"Parameter {param_a_name} not found in layer_a")
        raise AttributeError(f"Parameter {param_a_name} not found in layer_a")

    try:
        param_b = _get_module_by_path(layer_b, param_b_name)
    except AttributeError:
        logger.warning(f"Parameter {param_b_name} not found in layer_b")
        raise AttributeError(f"Parameter {param_b_name} not found in layer_b")


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
    q_name_a: str,
    k_name_a: str,
    v_name_a: str,
    o_name_a: str,
    q_name_b: str,
    k_name_b: str,
    v_name_b: str,
    o_name_b: str,
    backbone_attr: Optional[List[str]] = None
) -> nn.Module:
    """
    Initialize attention parameters of model_a using parameters from pretrained model_b.

    Args:
        model_a: Target model to be initialized
        model_b: Source pretrained model
        q_name_a, k_name_a, v_name_a, o_name_a: Names of attention parameters in model_a
        q_name_b, k_name_b, v_name_b, o_name_b: Names of attention parameters in model_b
        backbone_attr: List of attribute names to look for backbone

    Returns:
        Initialized model_a
    """
    # Create mapping dictionary for attention parameters
    mapping = {
        f"{q_name_a}": f"{q_name_b}",
        f"{k_name_a}": f"{k_name_b}",
        f"{v_name_a}": f"{v_name_b}",
        f"{o_name_a}": f"{o_name_b}"
    }

    return initialize_from_pretrained(model_a, model_b, mapping, backbone_attr)

def initialize_mlp_params(
    model_a: nn.Module,
    model_b: nn.Module,
    fc1_name_a: str,
    fc2_name_a: str,
    fc1_name_b: str,
    fc2_name_b: str,
    backbone_attr: Optional[List[str]] = None
) -> nn.Module:
    """
    Initialize MLP parameters of model_a using parameters from pretrained model_b.

    Args:
        model_a: Target model to be initialized
        model_b: Source pretrained model
        fc1_name_a, fc2_name_a: Names of MLP parameters in model_a
        fc1_name_b, fc2_name_b: Names of MLP parameters in model_b
        backbone_attr: List of attribute names to look for backbone

    Returns:
        Initialized model_a
    """
    # Create mapping dictionary with mlp prefix
    mapping = {
        f"{fc1_name_a}": f"{fc1_name_b}",
        f"{fc2_name_a}": f"{fc2_name_b}"
    }

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

