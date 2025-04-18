import os
import sys
import torch
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flazoo.models import DeltaNetVisionModel, TransformerVisionModel
from flazoo.helpers.initializer import (
    initialize_attention_params,
    initialize_mlp_params,
    initialize_custom_mapping
)

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize a model with parameters from a pretrained model")
    parser.add_argument("--target_model", type=str, default="deltanet", help="Target model type")
    parser.add_argument("--source_model", type=str, default="transformer", help="Source model type")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers for both models")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for both models")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--init_type", type=str, default="attention", 
                        choices=["attention", "mlp", "custom"], 
                        help="Type of initialization to perform")
    return parser.parse_args()

def create_model(model_type, num_layers, hidden_size, num_heads):
    """Create a model with the specified configuration."""
    if model_type == "deltanet":
        config = {
            "num_hidden_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "image_size": 224,
            "patch_size": 16,
        }
        return DeltaNetVisionModel(config)
    elif model_type == "transformer":
        config = {
            "num_hidden_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "image_size": 224,
            "patch_size": 16,
        }
        return TransformerVisionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    args = parse_args()
    
    # Create models
    print(f"Creating target model ({args.target_model})...")
    model_a = create_model(args.target_model, args.num_layers, args.hidden_size, args.num_heads)
    
    print(f"Creating source model ({args.source_model})...")
    model_b = create_model(args.source_model, args.num_layers, args.hidden_size, args.num_heads)
    
    # Initialize parameters
    if args.init_type == "attention":
        print("Initializing attention parameters...")
        # Standard attention parameter names
        initialize_attention_params(
            model_a=model_a,
            model_b=model_b,
            q_mapping="q_proj",
            k_mapping="k_proj",
            v_mapping="v_proj",
            o_mapping="o_proj"
        )
    
    elif args.init_type == "mlp":
        print("Initializing MLP parameters...")
        # Example MLP parameter mapping
        mlp_mapping = {
            "fc1": "fc1",
            "fc2": "fc2",
            "act": "act"
        }
        initialize_mlp_params(
            model_a=model_a,
            model_b=model_b,
            mlp_mapping=mlp_mapping
        )
    
    elif args.init_type == "custom":
        print("Initializing with custom parameter mapping...")
        # Example of custom parameter mapping
        custom_mapping = {
            "attn.q_proj": "attn.query_proj",
            "attn.k_proj": "attn.key_proj",
            "attn.v_proj": "attn.value_proj",
            "attn.o_proj": "attn.output_proj",
            "mlp.fc1": "ffn.linear1",
            "mlp.fc2": "ffn.linear2"
        }
        initialize_custom_mapping(
            model_a=model_a,
            model_b=model_b,
            param_mapping=custom_mapping
        )
    
    print("Initialization complete!")
    
    # Verify that parameters have been initialized
    print("\nVerifying parameter initialization...")
    # This is just a simple check - in a real scenario you would verify more thoroughly
    for name, param in model_a.named_parameters():
        if "attn" in name and args.init_type in ["attention", "custom"]:
            print(f"Attention parameter initialized: {name}, shape: {param.shape}")
            break
    
    for name, param in model_a.named_parameters():
        if "mlp" in name and args.init_type in ["mlp", "custom"]:
            print(f"MLP parameter initialized: {name}, shape: {param.shape}")
            break

if __name__ == "__main__":
    main()
