import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import math
from enum import Enum


class VisualizationStyle(Enum):
    """
    Preset visualization styles for attention maps.
    Each style defines a color scheme and visual parameters.
    """

    STANDARD = "standard"  # Standard viridis colormap
    HEAT = "heat"  # Heat map style with red-yellow gradient
    COOL = "cool"  # Cool blue-green gradient
    CONTRAST = "contrast"  # High contrast black and white
    SPECTRAL = "spectral"  # Rainbow-like spectral colormap
    ELEGANT = "elegant"  # Elegant black background with vibrant colors
    PASTEL = "pastel"  # Soft pastel colors
    DARK = "dark"  # Dark mode with bright highlights


class ModelVisualizer:
    """
    Wrapper class for visualizing internal outputs from FLA models.
    Captures attention outputs without modifying the model itself.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the visualizer with a model.

        Args:
            model: The FLA model to visualize
        """
        self.model = model
        self.hooks = []
        self.attention_outputs = {}
        self.target_layers = []
        self.target_batch_idx = None

    def register_attention_hooks(self, layer_indices: List[int]):
        """
        Register hooks for specific layers to capture attention outputs.

        Args:
            layer_indices: List of layer indices to capture attention from
        """
        # Clear any existing hooks
        self.remove_hooks()
        self.target_layers = layer_indices

        # Register new hooks
        for layer_idx in layer_indices:
            if not hasattr(self.model, "encoder") or not hasattr(
                self.model.encoder, "blocks"
            ):
                raise ValueError("Model structure not compatible with this visualizer")

            if layer_idx >= len(self.model.encoder.blocks):
                raise ValueError(
                    f"Layer index {layer_idx} out of range (model has {len(self.model.encoder.blocks)} layers)"
                )

            # Get the attention module
            block = self.model.encoder.blocks[layer_idx]
            if not hasattr(block, "attn"):
                raise ValueError(f"Layer {layer_idx} does not have an attention module")

            # Register the hook
            hook = block.attn.register_forward_hook(self._make_hook_fn(layer_idx))
            self.hooks.append(hook)

    def _make_hook_fn(self, layer_idx: int):
        """
        Create a hook function for a specific layer.

        Args:
            layer_idx: The index of the layer to hook

        Returns:
            A hook function
        """

        def hook_fn(module, input, output):
            # Most attention modules return (output, attentions, past_key_values)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                # Store the attention outputs
                if self.target_batch_idx is not None:
                    # Only store for the target batch index
                    if output[1].dim() == 4:  # [batch, heads, seq_len, seq_len]
                        self.attention_outputs[layer_idx] = (
                            output[1][self.target_batch_idx].detach().cpu()
                        )
                    else:
                        # Handle other attention output formats
                        self.attention_outputs[layer_idx] = output[1].detach().cpu()
                else:
                    # Store all batch items
                    self.attention_outputs[layer_idx] = output[1].detach().cpu()

        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_outputs = {}

    def set_target_batch(self, batch_idx: Optional[int] = None):
        """
        Set which batch item to visualize.

        Args:
            batch_idx: Index of the batch item to visualize, or None to visualize all
        """
        self.target_batch_idx = batch_idx

    def visualize_attention(
        self,
        input_data: torch.Tensor,
        output_dir: str = "attention_vis",
        filename_prefix: str = "attention",
        style: Union[str, VisualizationStyle] = VisualizationStyle.STANDARD,
        show_plots: bool = False,
        dpi: int = 300,
    ):
        """
        Run the model on input data and visualize the attention maps.

        Args:
            input_data: Input tensor to the model
            output_dir: Directory to save visualizations
            filename_prefix: Prefix for saved files
            style: Visualization style preset or custom colormap name
            show_plots: Whether to display plots in addition to saving them
            dpi: Resolution of saved images
        """
        # Clear previous attention outputs
        self.attention_outputs = {}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Run the model with hooks active
        with torch.no_grad():
            self.model(input_data, output_attentions=True)

        # Visualize the attention maps
        for layer_idx, attention in self.attention_outputs.items():
            if attention is None:
                print(f"No attention output captured for layer {layer_idx}")
                continue

            # Handle different attention shapes
            if attention.dim() == 3:  # [heads, seq_len, seq_len]
                self._visualize_single_attention(
                    attention,
                    layer_idx,
                    output_dir,
                    filename_prefix,
                    style,
                    show_plots,
                    dpi,
                )

            elif attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                # If we have a batch dimension but target_batch_idx wasn't set
                # Just use the first item in the batch
                batch_idx = (
                    0 if self.target_batch_idx is None else self.target_batch_idx
                )
                if batch_idx < attention.size(0):
                    self._visualize_single_attention(
                        attention[batch_idx],
                        layer_idx,
                        output_dir,
                        f"{filename_prefix}_batch{batch_idx}",
                        style,
                        show_plots,
                        dpi,
                    )
            else:
                print(f"Unsupported attention shape: {attention.shape}")

    def _visualize_single_attention(
        self,
        attention: torch.Tensor,
        layer_idx: int,
        output_dir: str,
        filename_prefix: str,
        style: Union[str, VisualizationStyle],
        show_plots: bool,
        dpi: int = 300,
    ):
        """
        Visualize attention for a single item.

        Args:
            attention: Attention tensor [heads, seq_len, seq_len]
            layer_idx: Layer index
            output_dir: Directory to save visualizations
            filename_prefix: Prefix for saved files
            style: Visualization style preset or custom colormap name
            show_plots: Whether to display plots
            dpi: Resolution of saved images
        """
        num_heads = attention.size(0)

        # Apply the selected style
        cmap, figsize, bg_color, text_color, edge_color, title_size, subtitle_size = (
            self._get_style_params(style)
        )

        # Set the style globally
        plt.style.use("dark_background" if bg_color == "black" else "default")

        # Create a grid of subplots
        grid_size = math.ceil(math.sqrt(num_heads))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=figsize, facecolor=bg_color
        )
        fig.tight_layout(pad=3.0)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Create a custom colormap if needed
        if isinstance(cmap, tuple):
            cmap = self._create_custom_colormap(*cmap)

        for head_idx in range(num_heads):
            if head_idx < len(axes):
                ax = axes[head_idx]
                # Normalize attention values for better visualization
                attn_data = attention[head_idx].numpy()
                im = ax.imshow(attn_data, cmap=cmap, interpolation="nearest")

                # Add a colorbar for each subplot if using elegant style
                if style == VisualizationStyle.ELEGANT or style == "elegant":
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

                # Style the subplot
                ax.set_title(
                    f"Head {head_idx}", color=text_color, fontsize=subtitle_size
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(edge_color)
                    spine.set_linewidth(1.5)

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis("off")

        # Add a main title
        plt.suptitle(
            f"Layer {layer_idx} Attention",
            color=text_color,
            fontsize=title_size,
            y=0.98,
        )

        # Save the figure
        filename = f"{filename_prefix}_layer{layer_idx}.png"
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=dpi,
            bbox_inches="tight",
            facecolor=bg_color,
        )

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Also save the average attention across all heads
        plt.figure(figsize=(10, 8), facecolor=bg_color)
        avg_attention = attention.mean(dim=0).numpy()
        im = plt.imshow(avg_attention, cmap=cmap, interpolation="bilinear")
        plt.title(
            f"Layer {layer_idx} - Average Attention",
            color=text_color,
            fontsize=title_size,
        )

        # Add a more stylish colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color=text_color)

        # Remove axis ticks for cleaner look
        plt.xticks([])
        plt.yticks([])

        # Add a subtle grid for the elegant style
        if style == VisualizationStyle.ELEGANT or style == "elegant":
            # Add grid lines at attention hotspots
            threshold = np.percentile(avg_attention, 90)
            hotspots = np.where(avg_attention > threshold)
            for i, j in zip(*hotspots):
                plt.axhline(
                    y=i, color=edge_color, linestyle="-", alpha=0.3, linewidth=0.5
                )
                plt.axvline(
                    x=j, color=edge_color, linestyle="-", alpha=0.3, linewidth=0.5
                )

        plt.tight_layout()

        avg_filename = f"{filename_prefix}_layer{layer_idx}_avg.png"
        plt.savefig(
            os.path.join(output_dir, avg_filename),
            dpi=dpi,
            bbox_inches="tight",
            facecolor=bg_color,
        )

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _get_style_params(self, style):
        """Get visualization parameters for the selected style.

        Args:
            style: A VisualizationStyle enum value or string

        Returns:
            Tuple of (colormap, figsize, bg_color, text_color, edge_color, title_size, subtitle_size)
        """
        # No need to import make_axes_locatable here as it's imported at the top

        # Convert string to enum if needed
        if isinstance(style, str):
            try:
                style = VisualizationStyle(style)
            except ValueError:
                # If it's not a valid enum value, treat it as a custom colormap name
                return style, (15, 15), "white", "black", "black", 18, 12

        # Define style parameters
        if style == VisualizationStyle.STANDARD:
            return "viridis", (15, 15), "white", "black", "black", 18, 12

        elif style == VisualizationStyle.HEAT:
            return "inferno", (15, 15), "black", "white", "#ff9500", 20, 14

        elif style == VisualizationStyle.COOL:
            return "cool", (15, 15), "white", "darkblue", "darkblue", 18, 12

        elif style == VisualizationStyle.CONTRAST:
            return "gray", (15, 15), "white", "black", "black", 18, 12

        elif style == VisualizationStyle.SPECTRAL:
            return "nipy_spectral", (16, 16), "#f5f5f5", "black", "gray", 20, 14

        elif style == VisualizationStyle.ELEGANT:
            return "magma", (18, 16), "black", "white", "#00b4d8", 22, 16

        elif style == VisualizationStyle.PASTEL:
            return (
                ("pastel", "white", "#ff9aa2", "#c7ceea"),
                (15, 15),
                "white",
                "#2e4057",
                "#2e4057",
                18,
                12,
            )

        elif style == VisualizationStyle.DARK:
            return "plasma", (16, 16), "#121212", "white", "#bb86fc", 20, 14

        else:  # Default
            return "viridis", (15, 15), "white", "black", "black", 18, 12

    def _create_custom_colormap(self, name, bg_color, start_color, end_color):
        """Create a custom colormap for special styles.

        Args:
            name: Name for the colormap
            bg_color: Background color
            start_color: Starting color for the gradient
            end_color: Ending color for the gradient

        Returns:
            A matplotlib colormap object
        """
        # Create a custom colormap for pastel style
        colors = [bg_color, start_color, end_color]
        n_bins = 256
        return mpl.colors.LinearSegmentedColormap.from_list(name, colors, N=n_bins)
