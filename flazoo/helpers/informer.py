import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
    }


def print_parameter_count(model: nn.Module) -> None:
    counts = get_parameter_count(model)

    print("=" * 50)
    print(f"Model Parameter Statistics:")
    print("-" * 40)
    print(f"Total Parameters:     {counts['total']:,}")
    print(f"Trainable Parameters: {counts['trainable']:,}")
    print(f"Frozen Parameters:    {counts['frozen']:,}")
    if counts["total"] > 0:
        print(
            f"Trainable Percentage: {counts['trainable'] / counts['total'] * 100:.2f}%"
        )
    print("=" * 50)


def get_parameter_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    param_info = {}

    for name, param in model.named_parameters():
        param_info[name] = {
            "shape": tuple(param.shape),
            "size": param.numel(),
            "trainable": param.requires_grad,
            "dtype": param.dtype,
            "device": param.device,
            "mean": param.data.mean().item() if param.numel() > 0 else 0,
            "std": param.data.std().item() if param.numel() > 0 else 0,
            "min": param.data.min().item() if param.numel() > 0 else 0,
            "max": param.data.max().item() if param.numel() > 0 else 0,
        }

    return param_info


def print_parameter_shapes(model: nn.Module, filter_str: Optional[str] = None) -> None:
    print("=" * 80)
    print(f"Model Parameter Shapes:")
    print("-" * 60)

    total_size = 0
    # Use natural sorting for consistent ordering
    for name, param in sorted(
        model.named_parameters(), key=lambda x: _natural_sort_key(x[0])
    ):
        if filter_str is None or filter_str in name:
            shape_str = str(tuple(param.shape))
            size = param.numel()
            total_size += size
            trainable = "✓" if param.requires_grad else "✗"
            print(f"{name:<50} {shape_str:<20} {size:>10,} {trainable}")

    print("-" * 60)
    print(f"Total Parameters: {total_size:,}")
    print("=" * 80)


def log_model_parameters(
    model: nn.Module, log_path: str = "model_parameters.log"
) -> None:
    """
    Get detailed parameter information from a model and save it to a log file in a nice format.
    The output shows the hierarchical structure of the model.

    Args:
        model: The PyTorch model to analyze
        log_path: Path to save the log file

    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True
    )

    # Open log file
    with open(log_path, "w") as f:
        # Write header
        f.write("=" * 100 + "\n")
        f.write(f"Model Parameter Information: {model.__class__.__name__}\n")
        f.write("=" * 100 + "\n\n")

        # Get overall parameter counts
        counts = get_parameter_count(model)
        f.write(f"Total Parameters:     {counts['total']:,}\n")
        f.write(f"Trainable Parameters: {counts['trainable']:,}\n")
        f.write(f"Frozen Parameters:    {counts['frozen']:,}\n")
        if counts["total"] > 0:
            f.write(
                f"Trainable Percentage: {counts['trainable'] / counts['total'] * 100:.2f}%\n"
            )
        f.write("\n" + "=" * 100 + "\n\n")

        # Build module hierarchy
        module_hierarchy = _build_module_hierarchy(model)

        # Write detailed parameter information with hierarchy
        f.write("Detailed Parameter Information:\n")
        f.write("-" * 100 + "\n\n")

        # Process modules in hierarchical order
        _write_module_hierarchy(model, module_hierarchy, f)


def _build_module_hierarchy(model: nn.Module) -> Dict[str, List[str]]:
    """
    Build a dictionary representing the module hierarchy.

    Args:
        model: The PyTorch model

    Returns:
        Dict mapping parent module names to lists of child module names
    """
    hierarchy = defaultdict(list)

    # Get all named modules
    for name, _ in model.named_modules():
        if name == "":  # Skip the root module
            continue

        # Add to parent's children list
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            hierarchy[parent_name].append(name)
        else:
            # Top-level modules are children of the root
            hierarchy[""].append(name)

    # Sort all lists using natural sorting
    for parent, children in hierarchy.items():
        hierarchy[parent] = sorted(children, key=_natural_sort_key)

    return hierarchy


def _natural_sort_key(s):
    """
    Helper function for natural sorting (e.g., layer1, layer2, layer10 instead of layer1, layer10, layer2).
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def _write_module_hierarchy(
    model: nn.Module,
    hierarchy: Dict[str, List[str]],
    file,
    parent_name: str = "",
    level: int = 0,
) -> None:
    """
    Recursively write module hierarchy with parameter information.

    Args:
        model: The PyTorch model
        hierarchy: Module hierarchy dictionary
        file: Open file to write to
        parent_name: Current parent module name
        level: Current hierarchy level
    """
    # Get children of this module
    children = hierarchy.get(parent_name, [])

    # Process each child - use natural sorting to handle numeric parts correctly
    for child_name in sorted(children, key=_natural_sort_key):
        # Get the module
        module = dict(model.named_modules())[child_name]

        # Calculate indentation
        indent = "  " * level

        # Write module information
        module_type = module.__class__.__name__
        file.write(f"{indent}● {child_name} ({module_type})\n")

        # Get parameters directly belonging to this module (not to its children)
        direct_params = {}
        prefix = child_name + "." if child_name else ""

        for name, param in model.named_parameters():
            # Check if parameter belongs directly to this module (not to a child module)
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                param_name = name[len(prefix) :] if prefix else name
                direct_params[param_name] = param

        # Write parameter information
        if direct_params:
            file.write(f"{indent}  Parameters:\n")

            # Calculate total parameters for this module
            module_total_params = sum(p.numel() for p in direct_params.values())
            module_trainable_params = sum(
                p.numel() for p in direct_params.values() if p.requires_grad
            )

            file.write(f"{indent}  - Total: {module_total_params:,} parameters ")
            file.write(f"({module_trainable_params:,} trainable, ")
            file.write(f"{module_total_params - module_trainable_params:,} frozen)\n")

            # Write individual parameter details
            for param_name, param in sorted(
                direct_params.items(), key=lambda x: _natural_sort_key(x[0])
            ):
                shape_str = str(tuple(param.shape))
                size = param.numel()
                trainable = "✓" if param.requires_grad else "✗"
                file.write(
                    f"{indent}  - {param_name}: {shape_str}, {size:,} elements, trainable: {trainable}\n"
                )

        # Process children recursively
        _write_module_hierarchy(model, hierarchy, file, child_name, level + 1)


def log_model_parameters_flat(
    model: nn.Module, log_path: str = "model_parameters_flat.log"
) -> None:
    """
    Get detailed parameter information from a model and save it to a log file in a flat format.
    This version doesn't show the hierarchical structure but lists all parameters in a table.

    Args:
        model: The PyTorch model to analyze
        log_path: Path to save the log file

    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True
    )

    # Open log file
    with open(log_path, "w") as f:
        # Write header
        f.write("=" * 100 + "\n")
        f.write(f"Model Parameter Information: {model.__class__.__name__}\n")
        f.write("=" * 100 + "\n\n")

        # Get overall parameter counts
        counts = get_parameter_count(model)
        f.write(f"Total Parameters:     {counts['total']:,}\n")
        f.write(f"Trainable Parameters: {counts['trainable']:,}\n")
        f.write(f"Frozen Parameters:    {counts['frozen']:,}\n")
        if counts["total"] > 0:
            f.write(
                f"Trainable Percentage: {counts['trainable'] / counts['total'] * 100:.2f}%\n"
            )
        f.write("\n" + "=" * 100 + "\n\n")

        # Write table header
        f.write(
            f"{'Parameter Name':<60} {'Shape':<20} {'Size':>12} {'Trainable':<10} {'Type':<10}\n"
        )
        f.write("-" * 100 + "\n")

        # Write parameter information - use natural sorting for consistent ordering
        for name, param in sorted(
            model.named_parameters(), key=lambda x: _natural_sort_key(x[0])
        ):
            shape_str = str(tuple(param.shape))
            size = param.numel()
            trainable = "✓" if param.requires_grad else "✗"
            dtype = str(param.dtype).split(".")[-1]
            f.write(
                f"{name:<60} {shape_str:<20} {size:>12,} {trainable:<10} {dtype:<10}\n"
            )
