# -*- coding: utf-8 -*-

import torch
from torch import nn
from typing import Optional
import math
import triton.language as tl
import einops
import triton
from einops import rearrange
from transformers.utils import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple, Union

"""
Cross Scan and Cross Merge implemented in Triton (only). Taken from https://github.com/MzeroMiko/VMamba/blob/main/classification/models/csm_triton.py
"""


@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor,  # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y: tl.tensor,  # (B, 4, C, H, W) | (B, H, W, 4, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_h = i_h * BH + tl.arange(0, BH)[:, None]
    pos_w = i_w * BW + tl.arange(0, BW)[None, :]
    neg_h = DH - i_h * BH - 1 - tl.arange(0, BH)[:, None]
    neg_w = DW - i_w * BW - 1 - tl.arange(0, BW)[None, :]
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h  # trans
        HWRoute2 = neg_h * DW + neg_w  # flip
        HWRoute3 = neg_w * DH + neg_h  # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w  # flip
        HWRoute3 = HWRoute2

    _tmp1 = DC * DH * DW

    y_ptr_base = (
        y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    )
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC

    if onebyone == 0:
        x_ptr_base = (
            x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        )
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = (
            x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        )
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(
                    p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw
                )
                tl.store(
                    p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw
                )
                tl.store(
                    p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw
                )
                tl.store(
                    p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw
                )
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        in_channel_first=True,
        out_channel_first=True,
        one_by_one=False,
        scans=0,
    ):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)

        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = (
            x.new_empty((B, 4, C, H * W))
            if out_channel_first
            else x.new_empty((B, H * W, 4, C))
        )
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(),
            y,
            (0 if in_channel_first else 1),
            (0 if out_channel_first else 1),
            0,
            (0 if not one_by_one else 1),
            scans,
            BC,
            BH,
            BW,
            C,
            H,
            W,
            NH,
            NW,
        )
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = (
                y.new_empty((B, 4, C, H, W))
                if in_channel_first
                else y.new_empty((B, H, W, 4, C))
            )
        else:
            x = (
                y.new_empty((B, C, H, W))
                if in_channel_first
                else y.new_empty((B, H, W, C))
            )

        triton_cross_scan_flex[(NH * NW, NC, B)](
            x,
            y.contiguous(),
            (0 if in_channel_first else 1),
            (0 if out_channel_first else 1),
            1,
            (0 if not one_by_one else 1),
            scans,
            BC,
            BH,
            BW,
            C,
            H,
            W,
            NH,
            NW,
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        in_channel_first=True,
        out_channel_first=True,
        one_by_one=False,
        scans=0,
    ):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = (
                y.new_empty((B, 4, C, H * W))
                if in_channel_first
                else y.new_empty((B, H * W, 4, C))
            )
        else:
            x = (
                y.new_empty((B, C, H * W))
                if in_channel_first
                else y.new_empty((B, H * W, C))
            )
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x,
            y.contiguous(),
            (0 if in_channel_first else 1),
            (0 if out_channel_first else 1),
            1,
            (0 if not one_by_one else 1),
            scans,
            BC,
            BH,
            BW,
            C,
            H,
            W,
            NH,
            NW,
        )
        return x

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = (
            x.new_empty((B, 4, C, H, W))
            if out_channel_first
            else x.new_empty((B, H, W, 4, C))
        )
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(),
            y,
            (0 if in_channel_first else 1),
            (0 if out_channel_first else 1),
            0,
            (0 if not one_by_one else 1),
            scans,
            BC,
            BH,
            BW,
            C,
            H,
            W,
            NH,
            NW,
        )
        return y, None, None, None, None, None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(
    x: torch.Tensor,
    in_channel_first=True,
    out_channel_first=True,
    one_by_one=False,
    scans=0,
    force_torch=False,
):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    assert x.is_cuda
    CSF = CrossScanTritonF
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(
    y: torch.Tensor,
    in_channel_first=True,
    out_channel_first=True,
    one_by_one=False,
    scans=0,
    force_torch=False,
):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    assert y.is_cuda
    CMF = CrossMergeTritonF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


@torch.compile
def multi_head_split_2d_torch(hidden_states: torch.Tensor, num_heads: int):
    """
    PyTorch implementation of multi-head scanning
    - Divides heads into 4 equal groups, each with a different scanning direction:
      - Group 1 (0 to n/4-1): Original sequence order
      - Group 2 (n/4 to n/2-1): 2D transpose
      - Group 3 (n/2 to 3n/4-1): Sequence reversal
      - Group 4 (3n/4 to n-1): 2D transpose + reversal

    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        num_heads: Number of attention heads (must be divisible by 4)

    Returns:
        Processed hidden states with different scanning patterns
    """
    B, L, D = hidden_states.shape
    head_dim = D // num_heads
    device = hidden_states.device
    dtype = hidden_states.dtype
    assert num_heads % 4 == 0, f"Number of heads {num_heads} must be divisible by 4"

    # Calculate dimensions for each group
    heads_per_group = num_heads // 4
    dim_per_group = heads_per_group * head_dim

    # Initialize result tensor
    result = torch.empty_like(hidden_states)

    # Group 1: Keep original order
    result[:, :, :dim_per_group] = hidden_states[:, :, :dim_per_group]

    # Prepare for 2D operations
    hw = int(math.sqrt(L))
    assert hw * hw == L, (
        f"Sequence length {L} must be a perfect square for 2D operations"
    )

    # Group 2: 2D transpose
    result[:, :, dim_per_group : 2 * dim_per_group] = einops.rearrange(
        hidden_states[:, :, dim_per_group : 2 * dim_per_group],
        "b (h w) d -> b (w h) d",
        h=hw,
        w=hw,
    )

    # Group 3: Sequence reversal
    result[:, :, 2 * dim_per_group : 3 * dim_per_group] = torch.flip(
        hidden_states[:, :, 2 * dim_per_group : 3 * dim_per_group], dims=[1]
    )

    # Group 4: 2D transpose + reversal
    result[:, :, 3 * dim_per_group :] = torch.flip(
        einops.rearrange(
            hidden_states[:, :, 3 * dim_per_group :],
            "b (h w) d -> b (w h) d",
            h=hw,
            w=hw,
        ),
        dims=[1],
    )

    return result


@torch.compile
def multi_head_merge_2d_torch(hidden_states: torch.Tensor, num_heads: int):
    """
    PyTorch implementation for merging results from multi-head scanning
    - Restores the original ordering of the hidden states after processing
    - Each group of heads is merged back to the original sequence order:
      - Group 1: Original order
      - Group 2: 2D transpose
      - Group 3: Sequence reversal
      - Group 4: 2D transpose + reversal

    Args:
        hidden_states: Processed hidden states of shape [batch_size, seq_len, hidden_size]
        num_heads: Number of attention heads (must be divisible by 4)

    Returns:
        Merged hidden states with original ordering restored
    """
    B, L, D = hidden_states.shape
    device = hidden_states.device
    head_dim = D // num_heads
    dtype = hidden_states.dtype
    assert num_heads % 4 == 0, f"Number of heads {num_heads} must be divisible by 4"

    # Calculate dimensions for each group
    heads_per_group = num_heads // 4
    dim_per_group = heads_per_group * head_dim

    # Initialize result tensor
    result = torch.empty_like(hidden_states)

    # Group 1: Keep original order
    result[:, :, :dim_per_group] = hidden_states[:, :, :dim_per_group]

    # Prepare for 2D operations
    hw = int(math.sqrt(L))

    # Group 2: Restore from 2D transpose
    result[:, :, dim_per_group : 2 * dim_per_group] = einops.rearrange(
        hidden_states[:, :, dim_per_group : 2 * dim_per_group],
        "b (w h) d -> b (h w) d",
        h=hw,
        w=hw,
    )

    # Group 3: Restore from sequence reversal
    result[:, :, 2 * dim_per_group : 3 * dim_per_group] = torch.flip(
        hidden_states[:, :, 2 * dim_per_group : 3 * dim_per_group], dims=[1]
    )

    # Group 4: Restore from 2D transpose + reversal
    result[:, :, 3 * dim_per_group :] = einops.rearrange(
        torch.flip(hidden_states[:, :, 3 * dim_per_group :], dims=[1]),
        "b (w h) d -> b (h w) d",
        h=hw,
        w=hw,
    )

    return result


# A wrapper function to handle both split and merge operations
# and to choose between torch and triton backends
# This function is designed to be universal and can be used for both splitting and merging operations


def multi_head_2d_scan(
    hidden_states: torch.Tensor,
    num_heads: int,
    backend: str = "torch",
    operation: str = "split",
    **kwargs,
) -> torch.Tensor:
    """
    Universal function for 2D multi-head scanning, supporting different backends and operations.

    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        num_heads: Number of attention heads (must be divisible by 4)
        head_dim: Dimension of each attention head
        backend: Which backend to use, currently supported: "torch", "triton" (if CUDA is available)
        operation: Which operation to perform, either "split" or "merge"
        **kwargs: Placeholder for additional parameters specific to the backend or operation
    Raises:
        ValueError: If the operation is not "split" or "merge"

    Returns:
        Processed hidden states with different scanning patterns or merged back
    """
    # Validate operation parameter
    valid_operations = ["split", "merge"]
    if operation not in valid_operations:
        raise ValueError(
            f"Operation must be one of {valid_operations}, got {operation}"
        )

    # Validate backend parameter
    valid_backends = ["torch", "triton"]
    if backend not in valid_backends:
        raise ValueError(f"Backend must be one of {valid_backends}, got {backend}")

    # Handle the triton backend
    if backend == "triton":
        raise NotImplementedError("Triton backend is not implemented yet.")
    # Handle the torch backend
    else:  # backend == "torch"
        if operation == "split":
            output = multi_head_split_2d_torch(hidden_states, num_heads)
        else:  # operation == "merge"
            output = multi_head_merge_2d_torch(hidden_states, num_heads)

    return output


@torch.compile
def multi_head_split_3d_torch(hidden_states: torch.Tensor, num_heads: int, canvas_thw: Tuple[int, int, int] = None):
    """
    PyTorch implementation of multi-head 3D scanning
    - Divides heads into 12 equal groups, each with a different scanning direction:
      - Groups 1-6: Six permutations of T, H, W dimensions (THW, TWH, HTW, HWT, WTH, WTH)
      - Groups 7-12: Same six permutations but with sequence reversal (flip)

    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        num_heads: Number of attention heads (must have at least 12 heads)

    Returns:
        Processed hidden states with different 3D scanning patterns
    """
    B, L, D = hidden_states.shape
    head_dim = D // num_heads
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # Calculate how many complete groups of 12 we can make
    complete_groups = (num_heads // 12) * 12
    heads_per_group = complete_groups // 12
    dim_per_group = heads_per_group * head_dim
    
    # Initialize result tensor
    result = torch.empty_like(hidden_states)
    
    if canvas_thw is not None:
        T, H, W = canvas_thw
    else:
        thw = int(round(L ** (1/3)))
        T, H, W = thw, thw, thw
    
    # Process only the complete groups (ignore remainder)
    if complete_groups > 0:
        # Group 1: Original order (T, H, W)
        result[:, :, :dim_per_group] = hidden_states[:, :, :dim_per_group]
        
        # Group 2: T, W, H permutation
        result[:, :, dim_per_group:2*dim_per_group] = einops.rearrange(
            hidden_states[:, :, dim_per_group:2*dim_per_group],
            "b (t h w) d -> b (t w h) d",
            t=T, h=H, w=W,
        )
        
        # Group 3: H, T, W permutation
        result[:, :, 2*dim_per_group:3*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 2*dim_per_group:3*dim_per_group],
            "b (t h w) d -> b (h t w) d",
            t=T, h=H, w=W,
        )
        
        # Group 4: H, W, T permutation
        result[:, :, 3*dim_per_group:4*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 3*dim_per_group:4*dim_per_group],
            "b (t h w) d -> b (h w t) d",
            t=T, h=H, w=W,
        )
        
        # Group 5: W, T, H permutation
        result[:, :, 4*dim_per_group:5*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 4*dim_per_group:5*dim_per_group],
            "b (t h w) d -> b (w t h) d",
            t=T, h=H, w=W,
        )
        
        # Group 6: W, H, T permutation
        result[:, :, 5*dim_per_group:6*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 5*dim_per_group:6*dim_per_group],
            "b (t h w) d -> b (w h t) d",
            t=T, h=H, w=W,
        )
        
        # Group 7: Original order (T, H, W) + flip
        result[:, :, 6*dim_per_group:7*dim_per_group] = torch.flip(
            hidden_states[:, :, 6*dim_per_group:7*dim_per_group], dims=[1]
        )
        
        # Group 8: T, W, H permutation + flip
        result[:, :, 7*dim_per_group:8*dim_per_group] = torch.flip(
            einops.rearrange(
                hidden_states[:, :, 7*dim_per_group:8*dim_per_group],
                "b (t h w) d -> b (t w h) d",
                t=T, h=H, w=W,
            ), dims=[1]
        )
        
        # Group 9: H, T, W permutation + flip
        result[:, :, 8*dim_per_group:9*dim_per_group] = torch.flip(
            einops.rearrange(
                hidden_states[:, :, 8*dim_per_group:9*dim_per_group],
                "b (t h w) d -> b (h t w) d",
                t=T, h=H, w=W,
            ), dims=[1]
        )
        
        # Group 10: H, W, T permutation + flip
        result[:, :, 9*dim_per_group:10*dim_per_group] = torch.flip(
            einops.rearrange(
                hidden_states[:, :, 9*dim_per_group:10*dim_per_group],
                "b (t h w) d -> b (h w t) d",
                t=T, h=H, w=W,
            ), dims=[1]
        )
        
        # Group 11: W, T, H permutation + flip
        result[:, :, 10*dim_per_group:11*dim_per_group] = torch.flip(
            einops.rearrange(
                hidden_states[:, :, 10*dim_per_group:11*dim_per_group],
                "b (t h w) d -> b (w t h) d",
                t=T, h=H, w=W,
            ), dims=[1]
        )
        
        # Group 12: W, H, T permutation + flip
        result[:, :, 11*dim_per_group:12*dim_per_group] = torch.flip(
            einops.rearrange(
                hidden_states[:, :, 11*dim_per_group:12*dim_per_group],
                "b (t h w) d -> b (w h t) d",
                t=T, h=H, w=W,
            ), dims=[1]
        )
    
    # Handle remaining heads (if any) - keep original order
    if complete_groups < num_heads:
        remaining_dim = (num_heads - complete_groups) * head_dim
        result[:, :, complete_groups*head_dim:] = hidden_states[:, :, complete_groups*head_dim:]
    
    return result


@torch.compile
def multi_head_merge_3d_torch(hidden_states: torch.Tensor, num_heads: int, canvas_thw: Tuple[int, int, int] = None):
    """
    PyTorch implementation for merging results from multi-head 3D scanning
    - Restores the original ordering of the hidden states after processing
    - Each group of heads is merged back to the original sequence order:
      - Groups 1-6: Reverse the six permutations of T, H, W dimensions
      - Groups 7-12: Reverse the flips and then the six permutations

    Args:
        hidden_states: Processed hidden states of shape [batch_size, seq_len, hidden_size]
        num_heads: Number of attention heads (must have at least 12 heads)

    Returns:
        Merged hidden states with original ordering restored
    """
    B, L, D = hidden_states.shape
    device = hidden_states.device
    head_dim = D // num_heads
    dtype = hidden_states.dtype
    
    # Calculate how many complete groups of 12 we can make
    complete_groups = (num_heads // 12) * 12
    heads_per_group = complete_groups // 12
    dim_per_group = heads_per_group * head_dim
    
    # Initialize result tensor
    result = torch.empty_like(hidden_states)
    

    if canvas_thw is not None:
        T, H, W = canvas_thw
    else:
        thw = int(round(L ** (1/3)))
        T, H, W = thw, thw, thw
    
    # Process only the complete groups (ignore remainder)
    if complete_groups > 0:
        # Group 1: Keep original order (T, H, W)
        result[:, :, :dim_per_group] = hidden_states[:, :, :dim_per_group]
        
        # Group 2: Restore from T, W, H permutation
        result[:, :, dim_per_group:2*dim_per_group] = einops.rearrange(
            hidden_states[:, :, dim_per_group:2*dim_per_group],
            "b (t w h) d -> b (t h w) d",
            t=T, w=W, h=H,
        )
        
        # Group 3: Restore from H, T, W permutation
        result[:, :, 2*dim_per_group:3*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 2*dim_per_group:3*dim_per_group],
            "b (h t w) d -> b (t h w) d",
            h=H, t=T, w=W,
        )
        
        # Group 4: Restore from H, W, T permutation
        result[:, :, 3*dim_per_group:4*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 3*dim_per_group:4*dim_per_group],
            "b (h w t) d -> b (t h w) d",
            h=H, w=W, t=T,
        )
        
        # Group 5: Restore from W, T, H permutation
        result[:, :, 4*dim_per_group:5*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 4*dim_per_group:5*dim_per_group],
            "b (w t h) d -> b (t h w) d",
            w=W, t=T, h=H,
        )
        
        # Group 6: Restore from W, H, T permutation
        result[:, :, 5*dim_per_group:6*dim_per_group] = einops.rearrange(
            hidden_states[:, :, 5*dim_per_group:6*dim_per_group],
            "b (w h t) d -> b (t h w) d",
            w=W, h=H, t=T,
        )
        
        # Group 7: Restore from original order (T, H, W) + flip
        result[:, :, 6*dim_per_group:7*dim_per_group] = torch.flip(
            hidden_states[:, :, 6*dim_per_group:7*dim_per_group], dims=[1]
        )
        
        # Group 8: Restore from T, W, H permutation + flip
        result[:, :, 7*dim_per_group:8*dim_per_group] = einops.rearrange(
            torch.flip(hidden_states[:, :, 7*dim_per_group:8*dim_per_group], dims=[1]),
            "b (t w h) d -> b (t h w) d",
            t=T, w=W, h=H,
        )
        
        # Group 9: Restore from H, T, W permutation + flip
        result[:, :, 8*dim_per_group:9*dim_per_group] = einops.rearrange(
            torch.flip(hidden_states[:, :, 8*dim_per_group:9*dim_per_group], dims=[1]),
            "b (h t w) d -> b (t h w) d",
            h=H, t=T, w=W,
        )
        
        # Group 10: Restore from H, W, T permutation + flip
        result[:, :, 9*dim_per_group:10*dim_per_group] = einops.rearrange(
            torch.flip(hidden_states[:, :, 9*dim_per_group:10*dim_per_group], dims=[1]),
            "b (h w t) d -> b (t h w) d",
            h=H, w=W, t=T,
        )
        
        # Group 11: Restore from W, T, H permutation + flip
        result[:, :, 10*dim_per_group:11*dim_per_group] = einops.rearrange(
            torch.flip(hidden_states[:, :, 10*dim_per_group:11*dim_per_group], dims=[1]),
            "b (w t h) d -> b (t h w) d",
            w=W, t=T, h=H,
        )
        
        # Group 12: Restore from W, H, T permutation + flip
        result[:, :, 11*dim_per_group:12*dim_per_group] = einops.rearrange(
            torch.flip(hidden_states[:, :, 11*dim_per_group:12*dim_per_group], dims=[1]),
            "b (w h t) d -> b (t h w) d",
            w=W, h=H, t=T,
        )
    
    # Handle remaining heads (if any) - keep original order
    if complete_groups < num_heads:
        remaining_dim = (num_heads - complete_groups) * head_dim
        result[:, :, complete_groups*head_dim:] = hidden_states[:, :, complete_groups*head_dim:]
    
    return result


def multi_head_3d_scan(
    hidden_states: torch.Tensor,
    num_heads: int,
    backend: str = "torch",
    operation: str = "split",
    canvas_thw: Tuple[int, int, int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Universal function for 3D multi-head scanning, supporting different backends and operations.

    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        num_heads: Number of attention heads (must have at least 12 heads)
        backend: Which backend to use, currently supported: "torch", "triton" (if CUDA is available)
        operation: Which operation to perform, either "split" or "merge"
        canvas_thw: Optional tuple specifying the dimensions of the 3D canvas (T, H, W)
        **kwargs: Placeholder for additional parameters specific to the backend or operation

    Raises:
        ValueError: If the operation is not "split" or "merge"

    Returns:
        Processed hidden states with different 3D scanning patterns or merged back
    """
    # Validate operation parameter
    valid_operations = ["split", "merge"]
    if operation not in valid_operations:
        raise ValueError(
            f"Operation must be one of {valid_operations}, got {operation}"
        )

    # Validate backend parameter
    valid_backends = ["torch", "triton"]
    if backend not in valid_backends:
        raise ValueError(f"Backend must be one of {valid_backends}, got {backend}")

    # Handle the triton backend
    if backend == "triton":
        raise NotImplementedError("Triton backend is not implemented yet.")
    # Handle the torch backend
    else:  # backend == "torch"
        if operation == "split":
            output = multi_head_split_3d_torch(hidden_states, num_heads, canvas_thw=canvas_thw)
        else:  # operation == "merge"
            output = multi_head_merge_3d_torch(hidden_states, num_heads, canvas_thw=canvas_thw)

    return output

class LearnableScan(nn.Module):
    def __init__(
        self, seq_len: int, temperature: float = 1.0, init_scale: float = 10.0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.temperature = temperature
        logits = torch.zeros(seq_len, seq_len)
        logits.fill_(-init_scale)
        logits.fill_diagonal_(init_scale)
        self.logits = nn.Parameter(logits)  # make the logits learnable

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Returns:
            Permuted tensor with the same shape as input
        """
        perm_matrix = F.gumbel_softmax(
            self.logits, tau=self.temperature, hard=True, dim=-1
        )  # (seq_len, seq_len)
        # TODO: this is very inefficient, and just serve for analysis
        x_permuted = torch.matmul(perm_matrix, x)  # (batch, seq_len, dim)

        return x_permuted