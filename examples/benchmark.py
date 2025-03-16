import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from flazoo import (
    ABCVisionConfig, ABCForImageClassification,
    BitNetVisionConfig, BitNetForImageClassification,
    DeltaNetVisionConfig, DeltaNetForImageClassification,
    GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification,
    GLAVisionConfig, GLAForImageClassification,
    GSAVisionConfig, GSAForImageClassification,
    HGRNVisionConfig, HGRNForImageClassification,
    HGRN2VisionConfig, HGRN2ForImageClassification,
    LightNetVisionConfig, LightNetForImageClassification,
    LinearAttentionVisionConfig, LinearAttentionForImageClassification,
    RetNetVisionConfig, RetNetForImageClassification,
    RWKV6VisionConfig, RWKV6ForImageClassification,
    TransformerVisionConfig, TransformerForImageClassification,
    NSAVisionConfig, NSAForImageClassification
)

MODEL_CONFIGS = {
    'deltanet': (DeltaNetVisionConfig, DeltaNetForImageClassification),
    'abc': (ABCVisionConfig, ABCForImageClassification),
    'gated_deltanet': (GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification),
    'bitnet': (BitNetVisionConfig, BitNetForImageClassification),
    'gla': (GLAVisionConfig, GLAForImageClassification),
    'gsa': (GSAVisionConfig, GSAForImageClassification),
    'hgrn': (HGRNVisionConfig, HGRNForImageClassification),
    'hgrn2': (HGRN2VisionConfig, HGRN2ForImageClassification),
    'lightnet': (LightNetVisionConfig, LightNetForImageClassification),
    'linear_attn': (LinearAttentionVisionConfig, LinearAttentionForImageClassification),
    'retnet': (RetNetVisionConfig, RetNetForImageClassification),
    'rwkv6': (RWKV6VisionConfig, RWKV6ForImageClassification),
    'transformer': (TransformerVisionConfig, TransformerForImageClassification),
    'nsa': (NSAVisionConfig, NSAForImageClassification)
}

def create_model(model_type: str, seq_len: int, hidden_size: int, num_layers: int, 
                heads: int, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """Create a model instance with the given configuration"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    ConfigClass, ModelClass = MODEL_CONFIGS[model_type]
    
    # Calculate image size and patch size to match the sequence length
    patch_size = 16
    image_size = int(np.sqrt(seq_len * (patch_size ** 2)))
    
    config = ConfigClass(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=heads,
        patch_size=patch_size,
        image_size=image_size,
        num_classes=1000,  # Standard ImageNet classes
        scan_type="uni-scan"
    )
    
    model = ModelClass(config).to(device=device, dtype=dtype)
    model.eval()
    return model

def generate_input(batch_size: int, seq_len: int, hidden_size: int, 
                  device: str, dtype: torch.dtype) -> torch.Tensor:
    """Generate random input tensors for benchmarking"""
    patch_size = 16
    channels = 3
    image_size = int(np.sqrt(seq_len * (patch_size ** 2)))
    
    # Create batch of random images
    return torch.randn(batch_size, channels, image_size, image_size, 
                      device=device, dtype=dtype)

def benchmark_model(model: torch.nn.Module, input_data: torch.Tensor, 
                  num_warmup: int = 10, num_runs: int = 50) -> Tuple[float, float]:
    """Benchmark model inference time"""
    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
    
    # Timed runs
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = 1.0 / avg_time
    
    return avg_time, throughput

def print_num_of_parameters(model_type, model):
    """Print the number of parameters in the model with nice format, two separate lines"""
    print("="*50)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {model_type}: {num_params:,}")
    print("="*50)

def run_benchmarks(model_types: List[str], seq_lengths: List[int], 
                  batch_size: int = 1, hidden_size: int = 256, 
                  num_layers: int = 6, heads: int = 16,
                  device: str = "cuda", dtype_str: str = "float32",
                  num_warmup: int = 10, num_runs: int = 50) -> Dict:
    """Run benchmarks across multiple models and sequence lengths"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[dtype_str]
    
    results = {
        "models": model_types,
        "seq_lengths": seq_lengths,
        "latency": {model_type: [] for model_type in model_types},
        "throughput": {model_type: [] for model_type in model_types}
    }
    
    for model_type in model_types:
        print(f"\nBenchmarking {model_type}...")
        for seq_len in tqdm(seq_lengths):
            try:
                # Create model for this sequence length
                model = create_model(
                    model_type, seq_len, hidden_size, num_layers, heads, device, dtype
                )

                print_num_of_parameters(model_type, model)
                
                # Generate input data
                inputs = generate_input(batch_size, seq_len, hidden_size, device, dtype)
                
                # Run benchmark
                latency, throughput = benchmark_model(
                    model, inputs, num_warmup=num_warmup, num_runs=num_runs
                )
                
                results["latency"][model_type].append(latency * 1000)  # Convert to ms
                results["throughput"][model_type].append(throughput)
                
                # Clean up to avoid CUDA OOM
                del model
                del inputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error benchmarking {model_type} at seq_len={seq_len}: {str(e)}")
                results["latency"][model_type].append(float('nan'))
                results["throughput"][model_type].append(float('nan'))
    
    return results

def plot_results(results: Dict, output_dir: str, plot_type: str = "latency"):
    """Plot benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for model_type in results["models"]:
        if plot_type == "latency":
            plt.plot(results["seq_lengths"], results["latency"][model_type], 
                    marker='o', label=model_type)
            plt.ylabel("Latency (ms)")
        else:
            plt.plot(results["seq_lengths"], results["throughput"][model_type], 
                    marker='o', label=model_type)
            plt.ylabel("Throughput (samples/sec)")
    
    plt.xlabel("Sequence Length")
    plt.title(f"{plot_type.capitalize()} vs Sequence Length")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Log scale for better visualization
    plt.xscale('log', base=2)
    if plot_type == "latency":
        plt.yscale('log', base=10)
    
    filename = os.path.join(output_dir, f"{plot_type}_vs_seqlen.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    # Save raw data
    data_file = os.path.join(output_dir, f"{plot_type}_data.npz")
    np.savez(data_file, 
             seq_lengths=results["seq_lengths"],
             **{f"{model_type}_{plot_type}": results[plot_type][model_type] 
                for model_type in results["models"]})
    print(f"Raw data saved to {data_file}")



def main():
    parser = argparse.ArgumentParser(description="Benchmark FLA models with varying sequence lengths")
    parser.add_argument("--models", nargs='+', default=["deltanet", "transformer", "gated_deltanet"],
                        help="List of models to benchmark")
    parser.add_argument("--seq-lengths", nargs='+', type=int, 
                        default=[16, 32, 64, 128, 256, 512, 1024, 2048],
                        help="List of sequence lengths to test")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for inference")
    parser.add_argument("--hidden-size", type=int, default=256, 
                        help="Hidden size of models")
    parser.add_argument("--num-layers", type=int, default=6, 
                        help="Number of layers in models")
    parser.add_argument("--heads", type=int, default=64, 
                        help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmarks on")
    parser.add_argument("--dtype", type=str, default="float16", 
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for models")
    parser.add_argument("--warmup", type=int, default=10, 
                        help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=50, 
                        help="Number of timed runs")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"Models to benchmark: {args.models}")
    print(f"Sequence lengths: {args.seq_lengths}")
    
    results = run_benchmarks(
        model_types=args.models,
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        heads=args.heads,
        device=args.device,
        dtype_str=args.dtype,
        num_warmup=args.warmup,
        num_runs=args.runs
    )
    
    # Plot results
    plot_results(results, args.output_dir, "latency")
    plot_results(results, args.output_dir, "throughput")
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
