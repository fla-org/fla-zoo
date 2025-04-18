import torch
import os
import sys
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flazoo.models import DeltaNetForImageClassification
from flazoo.helpers.visualizer import ModelVisualizer, VisualizationStyle


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention maps from a vision model")
    parser.add_argument("--model_name", type=str, default="deltanet", help="Model type to use")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="attention_vis", help="Directory to save visualizations")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2], help="Layer indices to visualize")
    parser.add_argument("--style", type=str, default="standard",
                        choices=[s.value for s in VisualizationStyle],
                        help="Visualization style to use")
    parser.add_argument("--show", action="store_true", help="Show plots in addition to saving them")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved images")
    return parser.parse_args()


def load_image(image_path, size=224):
    """Load and preprocess an image for the model."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Add batch dimension


def main():
    args = parse_args()

    # Load model
    print(f"Loading {args.model_name} model...")
    model = DeltaNetForImageClassification.from_pretrained(f"fla-org/{args.model_name}-vision")

    # Create visualizer
    visualizer = ModelVisualizer(model)

    # Register hooks for specified layers
    print(f"Registering hooks for layers {args.layers}...")
    visualizer.register_attention_hooks(args.layers)

    # Load and preprocess image
    print(f"Loading image from {args.image_path}...")
    image = load_image(args.image_path)

    # Visualize attention
    print(f"Visualizing attention maps with style '{args.style}' (saving to {args.output_dir})...")
    visualizer.visualize_attention(
        image,
        output_dir=args.output_dir,
        filename_prefix=os.path.splitext(os.path.basename(args.image_path))[0],
        style=args.style,
        show_plots=args.show,
        dpi=args.dpi
    )

    print(f"Available visualization styles: {[s.value for s in VisualizationStyle]}")

    # Clean up
    visualizer.remove_hooks()
    print("Done!")


if __name__ == "__main__":
    main()
