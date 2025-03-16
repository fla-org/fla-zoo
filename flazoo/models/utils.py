# -*- coding: utf-8 -*-

import math
import einops
import torch
from transformers.utils import logging
import warnings
import torch.nn as nn
from typing import TYPE_CHECKING, Optional, Tuple, Union
from .scan import RandomScanWithReorder, cross_scan_fn, cross_merge_fn

logger = logging.get_logger(__name__)
    
def prepare_hidden_states_for_scan(hidden_states: torch.Tensor, scan_type: str = "uni-scan", training: bool = True, random_reorder: bool = True, random_scan_module: Optional[RandomScanWithReorder] = None):
    # hidden_states shape should be: (B, L, D)
    if scan_type == "uni-scan":
        return hidden_states
    elif scan_type == "random-scan":
        if not random_reorder:
            L = hidden_states.size(1)
            random_idx = torch.randperm(L, device=hidden_states.device)
            hidden_states = hidden_states[:, random_idx, :]
            return hidden_states

        # random scan with reorder
        assert random_scan_module is not None, "Random scan module is not provided"
        hidden_states = random_scan_module(hidden_states, training=training)
        return hidden_states

    elif scan_type == "flip-scan":
        hidden_states = hidden_states.flip(-2)
        return hidden_states
    elif scan_type == "switch-scan":
        # post process instead of pre process
        return hidden_states
    elif scan_type == "bi-scan":
        flipped_hidden_states = hidden_states.flip(-2)
        hidden_states = torch.cat([hidden_states, flipped_hidden_states], dim=0)
        return hidden_states

    # cross-scan
    B, L, D  = hidden_states.shape
    hw = int(math.sqrt(L))
    assert (hw * hw == L) 
    hidden_states = einops.rearrange(hidden_states, "b (h w) d -> b h w d", h=hw, w=hw) # change the shape to feed to cross_scan
    hidden_states = cross_scan_fn(hidden_states, in_channel_first=False, out_channel_first=False, one_by_one=False, scans=0)
    hidden_states = einops.rearrange(hidden_states, "b l k d -> (b k) l d")
    return hidden_states

def prepare_hidden_states_for_merge(hidden_states: torch.Tensor, scan_type: str = "uni-scan", layer_idx: int = None, random_reorder: bool = True, random_scan_module: Optional[RandomScanWithReorder] = None):
    # hidden_states shape should be: (BK, L, D), K=2 for bi-scan, K=1 for uni-scan, K=4 for cross-scan
    if scan_type == "uni-scan" or (scan_type == "random-scan" and not random_reorder) or scan_type == "flip-scan":
        return hidden_states
    elif scan_type == "random-scan":
        # random-scan with reorder
        assert random_scan_module is not None, "Random scan module is not provided"
        assert random_scan_module.layer_idx == layer_idx, f"Layer index mismatch between random scan module and current layer, {random_scan_module.layer_idx} != {layer_idx}"
        hidden_states = random_scan_module.restore_order(hidden_states)
        return hidden_states
    elif scan_type == "bi-scan":
        B = hidden_states.shape[0] // 2
        hidden_states = hidden_states[:B] + hidden_states[B:]
        return hidden_states
    elif scan_type == "switch-scan":
        assert layer_idx is not None
        # if layeridx % 2 == 0, then flip, if layeridx % 2 == 1, first shape into 2d, then transpose, then flatten back to 1d sequence
        if layer_idx % 2 == 0:
            hidden_states = hidden_states.flip(-2)
        else:
            B, L, D = hidden_states.shape
            hw = int(math.sqrt(L))
            hidden_states = einops.rearrange(hidden_states, "b (h w) d -> b h w d", h=hw, w=hw)
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = einops.rearrange(hidden_states, "b w h d -> b (w h) d")
        
        return hidden_states

    B, L, D  = hidden_states.shape
    hw = int(math.sqrt(L))
    hidden_states = einops.rearrange(hidden_states, "(b k) (h w) d -> b h w k d", k=4, h=hw, w=hw)
    hidden_states = cross_merge_fn(hidden_states, in_channel_first=False, out_channel_first=False, one_by_one=False, scans=0)
    return hidden_states


# Test code
def test_memory_efficient_random_scan():
    print("=== Testing MemoryEfficientRandomScan ===")
    B, L, D = 16, 10, 3
    
    # Create test data
    hidden_states = torch.zeros(B, L, D)
    # Use vectorized operations to fill test data
    b_indices = torch.arange(B).view(B, 1, 1).expand(B, L, D)
    l_indices = torch.arange(L).view(1, L, 1).expand(B, L, D)
    hidden_states = b_indices * 10 + l_indices + 1
    
    print(f"Original input (shape {hidden_states.shape}):")
    print(hidden_states[:, :, 0])  # Show only first column
    
    # Create model
    model = RandomScanWithReorder()
    
    # Apply random scan
    reordered = model(hidden_states)
    print("\nAfter reordering:")
    print(reordered[:, :, 0])
    
    # Show stored parameters
    print("\nStored parameters for reconstruction:")
    print(f"Base permutation: {model.base_perm}")
    print(f"Shifts: {model.shifts}")
    
    # Restore order
    restored = model.restore_order(reordered)
    print("\nAfter restoring order:")
    print(restored[:, :, 0])
    
    # Validate restoration
    is_equal = torch.allclose(hidden_states, restored)
    print("\nValidation of restoration:")
    print(f"Original and restored data are equal: {is_equal}")
    
    if not is_equal:
        print(f"Maximum difference: {torch.max(torch.abs(hidden_states - restored))}")
    
    # Check that buffers were properly cleaned
    print("\nChecking if buffers were cleaned:")
    has_base_perm = hasattr(model, 'base_perm') and model.base_perm is not None
    has_shifts = hasattr(model, 'shifts') and model.shifts is not None
    print(f"Still has base_perm: {has_base_perm}")
    print(f"Still has shifts: {has_shifts}")

# Multiple random tests
def test_multiple_random_scan():
    print("\n=== Multiple Random Tests ===")
    B, L, D = 8, 16, 32
    model = RandomScanWithReorder()
    
    # Use pre-generated data to avoid loops
    all_test_data = torch.randn(5, B, L, D)
    
    for i in range(5):
        hidden_states = all_test_data[i]
        reordered = model(hidden_states)
        restored = model.restore_order(reordered)
        is_equal = torch.allclose(hidden_states, restored)
        max_diff = torch.max(torch.abs(hidden_states - restored)).item()
        print(f"Test {i+1}: Restoration correct: {is_equal}, Maximum difference: {max_diff:.8f}")

# Performance benchmark
def benchmark_test_random_scan():
    print("\n=== Performance Test ===")
    import time
    
    B, L, D = 32, 512, 768  
    hidden_states = torch.randn(B, L, D)
    model = RandomScanWithReorder()
    
    # Warm-up
    for _ in range(3):
        reordered = model(hidden_states)
        restored = model.restore_order(reordered)
    
    # Timing
    start = time.time()
    iterations = 10
    
    for _ in range(iterations):
        reordered = model(hidden_states)
        restored = model.restore_order(reordered)
    
    elapsed = time.time() - start
    print(f"Average time per forward+restore: {(elapsed / iterations) * 1000:.2f} ms")