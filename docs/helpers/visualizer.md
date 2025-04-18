# Attention Visualization 🎨

This module provides tools for visualizing attention patterns in FLA models. The visualizer captures attention outputs from specific layers without modifying the model itself.

## Features

- **Non-intrusive design**: Uses PyTorch hooks to capture attention outputs during forward passes
- **Flexible layer selection**: Visualize attention from any specified layers
- **Batch item selection**: Focus on a specific item in a batch
- **Comprehensive visualization**: Creates both per-head and averaged attention visualizations
- **Beautiful preset styles**: Choose from multiple visualization styles for aesthetically pleasing outputs

## Visualization Styles

The visualizer includes several preset styles to make your attention maps visually appealing:

| Style | Description | Best For |
|-------|-------------|----------|
| `standard` | Standard viridis colormap | General purpose |
| `heat` | Heat map style with red-yellow gradient | Highlighting intensity |
| `cool` | Cool blue-green gradient | Subtle patterns |
| `contrast` | High contrast black and white | Pattern recognition |
| `spectral` | Rainbow-like spectral colormap | Distinguishing ranges |
| `elegant` | Elegant black background with vibrant colors | Presentations |
| `pastel` | Soft pastel colors | Gentle visualization |
| `dark` | Dark mode with bright highlights | Low-light viewing |

## Usage

### Basic Example

```python
from flazoo.models import DeltaNetForImageClassification
from flazoo.helpers.visualizer import ModelVisualizer, VisualizationStyle

# Load model
model = DeltaNetForImageClassification.from_pretrained("fla-org/deltanet-vision")

# Create visualizer
visualizer = ModelVisualizer(model)

# Register hooks for layers 0, 1, and 2
visualizer.register_attention_hooks([0, 1, 2])

# Visualize attention with the elegant style
visualizer.visualize_attention(
    image, 
    output_dir="attention_vis",
    filename_prefix="example_image",
    style=VisualizationStyle.ELEGANT
)

# Clean up when done
visualizer.remove_hooks()
```

### Command Line Example

```bash
python examples/vision/visualize_attention.py \
    --image_path path/to/image.jpg \
    --layers 0 1 2 \
    --style elegant \
    --output_dir attention_maps
```

## Output

For each layer, the visualizer generates:

1. **Per-head visualization**: A grid showing attention patterns for each attention head
2. **Average attention**: A heatmap showing the average attention across all heads

## API Reference

### `ModelVisualizer`

```python
class ModelVisualizer:
    def __init__(self, model):
        """Initialize with a model."""
        
    def register_attention_hooks(self, layer_indices):
        """Register hooks for specific layers."""
        
    def set_target_batch(self, batch_idx=None):
        """Set which batch item to visualize."""
        
    def visualize_attention(self, input_data, output_dir="attention_vis", 
                           filename_prefix="attention", 
                           style=VisualizationStyle.STANDARD,
                           show_plots=False, dpi=300):
        """Run the model and visualize attention maps."""
        
    def remove_hooks(self):
        """Remove all registered hooks."""
```

### `VisualizationStyle`

```python
class VisualizationStyle(Enum):
    STANDARD = "standard"
    HEAT = "heat"
    COOL = "cool"
    CONTRAST = "contrast"
    SPECTRAL = "spectral"
    ELEGANT = "elegant"
    PASTEL = "pastel"
    DARK = "dark"
```
