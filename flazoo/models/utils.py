# -*- coding: utf-8 -*-

import math
import triton.language as tl
import einops
import triton
import torch
from einops import rearrange
from transformers.utils import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple, Union
from .scan import RandomScanWithReorder
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         unpad_input)
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

try:
    from native_sparse_attention.ops.parallel import parallel_nsa, parallel_nsa_compression
except ImportError:
    warnings.warn(
        "Native Sparse Attention is not installed. Please check the package installation.",
        category=ImportWarning
    )
    parallel_nsa = None
    parallel_nsa_compression = None

logger = logging.get_logger(__name__)


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

"""
Attention implementation used in hybrid model, adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/attn.py
"""

class VAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ):
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        o = flash_attn_func(
            q, k, v,
            causal=False, # use non-causal attention for vision
            window_size=(-1, -1)
        )
        o = o.reshape(batch_size, q_len, self.hidden_size)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None

class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention layer that encapsulates sparse attention mechanism from parallel_nsa. This is a non-causal layer.
    
    reference: 
    - paper: https://arxiv.org/abs/2502.11089
    - Triton Implementation: https://github.com/fla-org/native-sparse-attention
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        block_size: int = 64,
        window_size: int = 64,
        num_blocks: int = 16,
        layer_idx: int = None
    ):
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = (self.num_heads // 16)
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.scale = self.head_dim ** -0.5

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Gate projections for selective, window and compress attention
        self.g_cmp_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_slc_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_swa_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False) if window_size > 0 else None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_indices: Optional[torch.Tensor] = None,
        block_counts: Optional[Union[torch.LongTensor, int]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, D]
            block_indices (Optional[torch.Tensor]): Precomputed block indices for selective attention
            block_counts (Optional[Union[torch.LongTensor, int]]): Number of selected blocks per token
            output_attentions (bool): Whether to return attention weights
        """
        from native_sparse_attention.ops.parallel import parallel_nsa_with_compression

        if parallel_nsa_with_compression is None:
            raise ImportError("Native Sparse Attention is not installed. Please check the package installation.")

        batch_size, seq_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Compute gate scores
        g_cmp = torch.sigmoid(self.g_cmp_proj(hidden_states))
        g_slc = torch.sigmoid(self.g_slc_proj(hidden_states))
        g_swa = torch.sigmoid(self.g_swa_proj(hidden_states)) if self.window_size > 0 else torch.zeros_like(g_slc)
        
        # Use block_counts if provided, otherwise use default num_blocks
        if block_counts is None:
            assert seq_len % self.block_size == 0, f"Sequence length must be divisible by block size, got {seq_len} and {self.block_size}"
            block_counts = seq_len // self.block_size
            
        # Apply native sparse attention with compression
        assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

        attn_output, block_indices = parallel_nsa_with_compression(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_counts=block_counts,
            block_size=self.block_size,
            window_size=self.window_size,
            scale=self.scale,
            head_first=False
        )
        
        # Project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            attentions = None  # TODO: Not implemented
        else:
            attentions = None

        return attn_output, attentions, None
