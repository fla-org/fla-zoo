import torch
import einops
import math
import matplotlib.pyplot as plt
import numpy as np

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

def window_partition_simple(x, window_size):
    """Implementation with the exact einops rearrange from the selected code"""
    # First reshape to B, D, H*W and then permute to B, H*W, D
    B, C, H, W = x.shape
    hw = H  # Assuming H = W
    x_reshaped = x.flatten(2).transpose(1, 2) # (B, H*W, D)
    # Now apply the exact rearrange operation from the selected code
    simple_windows = einops.rearrange(
        x_reshaped, 
        "b (x wx y wy) d -> (b x y) (wx wy) d", 
        wx=window_size, 
        wy=window_size, 
        x=hw//window_size, 
        y=hw//window_size
    )
    return simple_windows

def window_reverse_simple(windows, window_size, H, W):
    """Implementation with the exact einops rearrange from the selected code"""
    # First reshape to B, D, H*W and then permute to B, H*W, D
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    hw = H  # Assuming H = W
    simple_windows = einops.rearrange(
        windows, 
        "(b x y) (wx wy) d -> b d (x wx) (y wy)", 
        wx=window_size, 
        wy=window_size, 
        x=hw//window_size, 
        y=hw//window_size
    )
    return simple_windows

B, L, D = 2, 256, 3
hw = int(math.sqrt(L))  # hw = 16
window_size = 4

x = torch.randn(B, D, hw, hw)   
windows = window_partition(x, window_size)
simple_windows = window_partition_simple(x, window_size)
print(windows.shape)  # (B * hw // window_size * hw // window_size, window_size * window_size, D)
print(simple_windows.shape)
# test identical
assert torch.allclose(windows, simple_windows)
print("Test passed!")

reverse_windows = window_reverse(windows, window_size, hw, hw)
simple_reverse_windows = window_reverse_simple(simple_windows, window_size, hw, hw)
print(reverse_windows.shape)
print(simple_reverse_windows.shape)
# test identical
assert torch.allclose(reverse_windows, simple_reverse_windows)
print("Test passed!")