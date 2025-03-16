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

"""
Cross Scan and Cross Merge implemented in Triton (only). Taken from https://github.com/MzeroMiko/VMamba/blob/main/classification/models/csm_triton.py
"""

@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor, # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y: tl.tensor, # (B, 4, C, H, W) | (B, H, W, 4, C)
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

    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h # trans
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = neg_w * DH + neg_h # trans + flip
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
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = HWRoute2      

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
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
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
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
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
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
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
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
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
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

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BH, BW, C, H, W, NH, NW
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
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
        
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
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
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
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
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    assert x.is_cuda
    CSF = CrossScanTritonF
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    assert y.is_cuda
    CMF = CrossMergeTritonF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
