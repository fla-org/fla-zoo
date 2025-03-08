import torch
from torch import nn

class BatchRandomScan(nn.Module):
    """Fully loop-free batch random scanning - using random shift method"""
    
    def __init__(self):
        super().__init__()
        self.register_buffer('last_indices', None, persistent=False)
    
    def forward(self, hidden_states, training=True):
        # hidden_states: [B, L, D]
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # Generate a single base permutation
        base_perm = torch.randperm(L, device=device)
        
        # Use random tensor to generate different shifts for each batch
        # Generate random integers in range [0, L-1] as shift values
        shifts = torch.randint(0, L, (B,), device=device)
        
        # Apply different circular shifts for each batch, creating B different permutations
        # Use vectorized operations instead of loops
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        shifted_indices = (indices + shifts.unsqueeze(1)) % L
        
        # Use shifted indices to reorder the base permutation
        random_indices = base_perm.unsqueeze(0).expand(B, L).gather(1, shifted_indices)
        
        # Store indices for restoring order
        self.last_indices = random_indices
        
        # Apply random ordering using advanced indexing
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, L)
        reordered = hidden_states[batch_indices, random_indices]
        
        return reordered
    
    def restore_order(self, hidden_states):
        # hidden_states: [B, L, D]
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        if self.last_indices is None:
            return hidden_states
        
        # Create reverse indices - using vectorized operations
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        # Use scatter to build reverse indices
        reverse_indices = torch.zeros_like(self.last_indices)
        reverse_indices.scatter_(1, self.last_indices, positions)
        
        # Apply reverse ordering
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, L)
        restored = hidden_states[batch_indices, reverse_indices]
        
        return restored

# Test code
def test_batch_random_scan():
    print("=== Testing BatchRandomScan ===")
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
    model = BatchRandomScan()
    
    # Apply random scan
    reordered = model(hidden_states)
    print("\nAfter reordering:")
    print(reordered[:, :, 0])
    
    # Show random indices
    print("\nRandom indices used:")
    print(model.last_indices)
    
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

# Multiple random tests
def test_multiple_random():
    print("\n=== Multiple Random Tests ===")
    B, L, D = 8, 16, 32
    model = BatchRandomScan()
    
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
    model = BatchRandomScan()
    
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

# Run tests
if __name__ == "__main__":
    test_batch_random_scan()
    test_multiple_random()
    benchmark_test()