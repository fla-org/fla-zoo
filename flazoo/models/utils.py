# -*- coding: utf-8 -*-

import math
import einops
import torch
from transformers.utils import logging
import warnings
import torch.nn as nn
from typing import TYPE_CHECKING, Optional, Tuple, Union
from .scan import cross_scan_fn, cross_merge_fn

logger = logging.get_logger(__name__)
    
def prepare_hidden_states_for_scan(hidden_states: torch.Tensor, train_scan_type: str = "uni-scan", test_scan_type : str = "uni-scan", training: bool = True, random_level: str = "sample", scan_module: Optional[nn.Module] = None):
    # radnom level should be "sample" or "batch"
    assert random_level in ["sample", "batch"], "random_level should be 'sample' or 'batch'"
    scan_type = train_scan_type if training else test_scan_type
    # hidden_states shape should be: (B, L, D)
    if scan_type == "uni-scan":
        return hidden_states
    elif scan_type == "random-scan":
        if random_level == "batch":
            L = hidden_states.size(1)
            random_idx = torch.randperm(L, device=hidden_states.device)
            hidden_states = hidden_states[:, random_idx, :]
        else:
            # random shuffle for each sample
            B, L, D = hidden_states.shape
            device = hidden_states.device
            random_indices = torch.argsort(torch.rand(B, L, device=device), dim=1)
            batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, L)
            hidden_states = hidden_states[batch_indices, random_indices]
        return hidden_states
    
    elif scan_type == "flip-scan":
        hidden_states = hidden_states.flip(-2)
        return hidden_states
    elif scan_type == "1d-shift-scan":
        return hidden_states
    elif scan_type == "2d-shift-scan":
        return hidden_states
    elif scan_type == "switch-scan":
        # post process instead of pre process
        return hidden_states
    elif scan_type == "bi-scan":
        flipped_hidden_states = hidden_states.flip(-2)
        hidden_states = torch.cat([hidden_states, flipped_hidden_states], dim=0)
        return hidden_states
    
    elif scan_type == "learnable-scan":
        assert scan_module is not None, "scan_module should be provided for learnable-scan"
        return scan_module(hidden_states)

    # cross-scan
    B, L, D  = hidden_states.shape
    hw = int(math.sqrt(L))
    assert (hw * hw == L) 
    hidden_states = einops.rearrange(hidden_states, "b (h w) d -> b h w d", h=hw, w=hw) # change the shape to feed to cross_scan
    hidden_states = cross_scan_fn(hidden_states, in_channel_first=False, out_channel_first=False, one_by_one=False, scans=0)
    hidden_states = einops.rearrange(hidden_states, "b l k d -> (b k) l d")
    return hidden_states

def prepare_hidden_states_for_merge(hidden_states: torch.Tensor, train_scan_type: str = "uni-scan", test_scan_type : str = "uni-scan", training: bool = True, layer_idx: Optional[int] = None):
    scan_type = train_scan_type if training else test_scan_type    
    # hidden_states shape should be: (BK, L, D), K=2 for bi-scan, K=1 for uni-scan, K=4 for cross-scan
    if scan_type == "uni-scan" or scan_type == "random-scan" or scan_type == "flip-scan" or scan_type == "learnable-scan":
        return hidden_states
    elif scan_type == "1d-shift-scan":
        B, L, D = hidden_states.shape
        shift = layer_idx
        hidden_states = torch.roll(hidden_states, shifts=shift, dims=1)
        return hidden_states
    elif scan_type == "2d-shift-scan":
        B, L, D = hidden_states.shape
        # back to 2d
        hw = int(math.sqrt(L))
        hidden_states = einops.rearrange(hidden_states, "b (h w) d -> b h w d", h=hw, w=hw)
        shift = int(math.sqrt(layer_idx))
        hidden_states = torch.roll(hidden_states, shifts=(shift, shift), dims=(1, 2))
        hidden_states = einops.rearrange(hidden_states, "b h w d -> b (h w) d")
        return hidden_states
    elif scan_type == "bi-scan":
        B = hidden_states.shape[0] // 2
        hidden_states = hidden_states[:B] + hidden_states[B:]
        return hidden_states
    elif scan_type == "switch-scan":
        assert layer_idx is not None, "layer_idx should be provided for switch-scan"
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

    # cross scan

    B, L, D  = hidden_states.shape
    hw = int(math.sqrt(L))
    hidden_states = einops.rearrange(hidden_states, "(b k) (h w) d -> b h w k d", k=4, h=hw, w=hw)
    hidden_states = cross_merge_fn(hidden_states, in_channel_first=False, out_channel_first=False, one_by_one=False, scans=0)
    return hidden_states

"""
Copied from https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_efficient.py
Huge thanks to MoonshotAI for their great work!
"""

def _calc_chunks(cu_seqlen, block_size):

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (block_size - 1)) // block_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), block_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * block_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return cu_chunk

"""
A simple mean pooling function, the mean is obtained within a chunk (or block) and the block is calculated by _calc_chunks
This is used in compressed flash linear attention
"""

def compress_seq(seq: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Compress sequence by mean pooling within chunks, assuming L is divisible by block_size
    Args:
        seq: input sequence with shape [B, L, D]
        block_size: size of each chunk/block
    Returns:
        compressed sequence with shape [B, L/block_size, D]
    """
    B, L, D = seq.shape
    assert L % block_size == 0, f"Sequence length {L} must be divisible by block_size {block_size}"
    
    # Reshape to [B, num_blocks, block_size, D] and compute mean
    num_blocks = L // block_size
    return seq.view(B, num_blocks, block_size, D).mean(dim=2)

def decompress_seq(compressed_seq: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Decompress sequence by repeating each compressed token block_size times
    Args:
        compressed_seq: input sequence with shape [B, L/block_size, D]
        block_size: size of each chunk/block
    Returns:
        decompressed sequence with shape [B, L, D]
    """
    B, num_blocks, D = compressed_seq.shape
    
    # First unsqueeze to add the block dimension
    # [B, num_blocks, 1, D]
    expanded = compressed_seq.unsqueeze(2)
    
    # Repeat along the block dimension
    # [B, num_blocks, block_size, D]
    repeated = expanded.expand(-1, -1, block_size, -1)
    
    # Reshape back to original sequence shape
    # [B, L, D]
    return repeated.reshape(B, num_blocks * block_size, D)