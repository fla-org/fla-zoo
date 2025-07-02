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


def log_model(model: nn.Module, log_path: str = None, log_config: bool = False) -> None:
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
        if hasattr(model, "config") and log_config:
            f.write("=" * 100 + "\n")
            f.write(f"Number of Layers: {model.config.num_hidden_layers}\n")
            f.write(f"Hidden Size: {model.config.hidden_size}\n")
            f.write(f"Number of Heads: {model.config.num_heads}\n")
            f.write("=" * 100 + "\n")
            f.write(f"FLA Attn Type: {model.config.fla_attn_type}\n")
            if model.config.attn is not None:
                f.write("=" * 100 + "\n")
                f.write(f"This is a Hybrid Model!\n")
                f.write("=" * 100 + "\n")
                f.write(f"Hybrid Attn Type: {model.config.attn_type}\n")
            else:
                f.write("=" * 100 + "\n")
                f.write(f"This is a pure FLA Model!\n")
                f.write("=" * 100 + "\n")
            if model.config.attn is not None:
                f.write("=" * 100 + "\n")
                # attn is a dict, write in a nice format
                f.write("Hybrid Attn Config:\n")
                for key, value in model.config.attn.items():
                    f.write(f"  {key}: {value}\n")

        f.write("=" * 100 + "\n\n")

        counts = get_parameter_count(model)
        f.write(f"Total Parameters:     {counts['total']:,}\n")
        f.write(f"Trainable Parameters: {counts['trainable']:,}\n")
        f.write(f"Frozen Parameters:    {counts['frozen']:,}\n")
        if counts["total"] > 0:
            f.write(
                f"Trainable Percentage: {counts['trainable'] / counts['total'] * 100:.2f}%\n"
            )
        f.write("\n" + "=" * 150 + "\n\n")

        f.write(
            f"{'Parameter Name':<60} {'Shape':<20} {'Size':>12} {'Trainable':>10} {'Type':>10}\n"
        )
        f.write("-" * 130 + "\n")

        for name, param in sorted(
            model.named_parameters(), key=lambda x: _natural_sort_key(x[0])
        ):
            shape_str = str(tuple(param.shape))
            size = param.numel()
            trainable = "✓" if param.requires_grad else "✗"
            dtype = str(param.dtype).split(".")[-1]
            f.write(
                f"{name:<60} {shape_str:<20} {size:>12,} {trainable:>6} {dtype:>15}\n"
            )


def log_model_with_emoji(model: nn.Module, log_path: str = None, log_config: bool = False) -> None:
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
        f.write("🎯" + "=" * 98 + "🎯\n")
        f.write(f"📊 Model Parameter Information: {model.__class__.__name__} 🚀\n")
        if hasattr(model, "config") and log_config:
            f.write("🎯" + "=" * 98 + "🎯\n")
            f.write(f"🏗️  Number of Layers: {model.config.num_hidden_layers}\n")
            f.write(f"📏 Hidden Size: {model.config.hidden_size}\n")
            f.write(f"🧠 Number of Heads: {model.config.num_heads}\n")
            f.write("⚡" + "=" * 98 + "⚡\n")
            f.write(f"🔥 FLA Attn Type: {model.config.fla_attn_type}\n")
            if model.config.attn is not None:
                f.write("🎉" + "=" * 98 + "🎉\n")
                f.write(f"🌟 This is a Hybrid Model! 😀🎊\n")
                f.write("🎉" + "=" * 98 + "🎉\n")
                f.write(f"🔀 Hybrid Attn Type: {model.config.attn_type}\n")
            else:
                f.write("💪" + "=" * 98 + "💪\n")
                f.write(f"⚡ This is a pure FLA Model! 🚀✨\n")
                f.write("💪" + "=" * 98 + "💪\n")
            if model.config.attn is not None:
                f.write("⚙️" + "=" * 98 + "⚙️\n")
                f.write("🔧 Hybrid Attn Config:\n")
                for key, value in model.config.attn.items():
                    f.write(f"  🔸 {key}: {value}\n")

        f.write("📈" + "=" * 98 + "📈\n\n")

        counts = get_parameter_count(model)
        f.write(f"🎯 Total Parameters:     {counts['total']:,} 🔢\n")
        f.write(f"🎓 Trainable Parameters: {counts['trainable']:,} ✅\n")
        f.write(f"🧊 Frozen Parameters:    {counts['frozen']:,} ❄️\n")
        if counts["total"] > 0:
            percentage = counts["trainable"] / counts["total"] * 100
            emoji = "🔥" if percentage > 90 else "⚡" if percentage > 50 else "🌟"
            f.write(f"📊 Trainable Percentage: {percentage:.2f}% {emoji}\n")
        f.write("\n" + "🎨" + "=" * 150 + "🎨\n\n")

        f.write(
            f"{'📋 Parameter Name':<65} {'📐 Shape':<25} {'📊 Size':>15} {'✅ Trainable':>20} {'🏷️  Type':>15}\n"
        )
        f.write("✨" + "-" * 150 + "✨\n")

        for name, param in sorted(
            model.named_parameters(), key=lambda x: _natural_sort_key(x[0])
        ):
            shape_str = str(tuple(param.shape))
            size = param.numel()
            trainable = "🟢" if param.requires_grad else "🔴"
            dtype = str(param.dtype).split(".")[-1]
            f.write(
                f"{name:<65} {shape_str:<25} {size:>15,} {trainable:>20} {dtype:>20}\n"
            )
