import torch.nn as nn
import os
import re
from typing import Dict

def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
    }

def _natural_sort_key(s):
    """
    Helper function for natural sorting (e.g., layer1, layer2, layer10 instead of layer1, layer10, layer2).
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def log_model(
    model: nn.Module, log_path: str = None
) -> None:
    """
    Get detailed parameter information from a model and save it to a log file in a flat format.

    Args:
        model: The model to analyze
        log_path: Path to save the log file, defaults to "{model_name}_info.log" in the current directory.
    """
    if log_path is None:
        log_path = f"{model.__class__.__name__}_info.log"

    os.makedirs(
        os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True
    )

    with open(log_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Model Parameter Information: {model.__class__.__name__}\n")
        f.write("=" * 100 + "\n\n")

        counts = get_parameter_count(model)
        f.write(f"Total Parameters:     {counts['total']:,}\n")
        f.write(f"Trainable Parameters: {counts['trainable']:,}\n")
        f.write(f"Frozen Parameters:    {counts['frozen']:,}\n")
        if counts["total"] > 0:
            f.write(
                f"Trainable Percentage: {counts['trainable'] / counts['total'] * 100:.2f}%\n"
            )
        f.write("\n" + "=" * 100 + "\n\n")

        f.write(
            f"{'Parameter Name':<60} {'Shape':<20} {'Size':>12} {'Trainable':<10} {'Type':<10}\n"
        )
        f.write("-" * 100 + "\n")

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
