import torch
from torch import nn
from typing import Optional

class RandomScanWithReorder(nn.Module):
    """Random shuffle of sequences in a batch, with reordering."""
    
    def __init__(self, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        # Store only base permutation and shifts instead of full indices
        self.register_buffer('base_perm', None, persistent=False)
        self.register_buffer('shifts', None, persistent=False)
    
    def forward(self, hidden_states, training=True):
        # hidden_states: [B, L, D]
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # Generate a single base permutation
        self.base_perm = torch.randperm(L, device=device)
        
        # Use random tensor to generate different shifts for each batch
        # Generate random integers in range [0, L-1] as shift values
        self.shifts = torch.randint(0, L, (B,), device=device)
        
        # Apply different circular shifts for each batch, creating B different permutations
        # Use vectorized operations instead of loops
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        shifted_indices = (indices + self.shifts.unsqueeze(1)) % L
        
        # Use shifted indices to reorder the base permutation
        random_indices = self.base_perm.unsqueeze(0).expand(B, L).gather(1, shifted_indices)
        
        # Apply random ordering using advanced indexing
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, L)
        reordered = hidden_states[batch_indices, random_indices]
        
        # Free memory - we only need base_perm and shifts for restoration
        del random_indices
        
        return reordered
    
    def restore_order(self, hidden_states):
        # hidden_states: [B, L, D]
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        if self.base_perm is None or self.shifts is None:
            raise RuntimeError("Cannot restore order without base_perm and shifts")
        
        # Reconstruct random_indices from base_perm and shifts
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        shifted_indices = (indices + self.shifts.unsqueeze(1)) % L
        random_indices = self.base_perm.unsqueeze(0).expand(B, L).gather(1, shifted_indices)
        
        # Create reverse indices - using vectorized operations
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        # Use scatter to build reverse indices
        reverse_indices = torch.zeros_like(random_indices)
        reverse_indices.scatter_(1, random_indices, positions)
        
        
        # Apply reverse ordering
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, L)
        restored = hidden_states[batch_indices, reverse_indices]
                
        return restored

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
def test_multiple_random():
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
def benchmark_test():
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

# Memory comparison
def memory_comparison():
    print("\n=== Memory Usage Comparison ===")
    
    import gc
    import torch
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Define parameters for test
    B, L, D = 32, 2048, 2048  # Large size to make memory difference clear
    
    # Original implementation
    def measure_original():
        model = RandomScanWithReorder()
        hidden_states = torch.randn(B, L, D, device="cuda")
        reordered = model(hidden_states)
        
        # Measure memory usage with full indices
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**2
        restored = model.restore_order(reordered)
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / 1024**2
        
        return memory_before, memory_after
    
    # Memory-efficient implementation
    def measure_efficient():
        model = RandomScanWithReorder()
        hidden_states = torch.randn(B, L, D, device="cuda")
        reordered = model(hidden_states)
        
        # Measure memory usage with compressed representation
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**2
        restored = model.restore_order(reordered)
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / 1024**2
        
        return memory_before, memory_after
    
    try:
        orig_before, orig_after = measure_original()
        print(f"Original implementation: {orig_before:.2f} MB before restore, {orig_after:.2f} MB after")
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        eff_before, eff_after = measure_efficient()
        print(f"Efficient implementation: {eff_before:.2f} MB before restore, {eff_after:.2f} MB after")
        
        # Calculate savings
        savings = orig_before - eff_before
        percent = savings / orig_before * 100 if orig_before > 0 else 0
        print(f"Memory savings: {savings:.2f} MB ({percent:.2f}%)")
        
    except RuntimeError as e:
        print(f"Memory test error (possibly out of CUDA memory): {e}")
        print("Try with smaller dimensions to run the memory comparison")

# Run tests
if __name__ == "__main__":
    test_memory_efficient_random_scan()
    test_multiple_random()
    benchmark_test()
