
# -*- coding: utf-8 -*-

import triton
import torch
from torch import IntTensor, BoolTensor
import warnings
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import math

try:
    from moba.moba_efficient import moba_attn_varlen
except ImportError:
    warnings.warn(
        "MoBA is not installed. Please install it from https://github.com/MoonshotAI/MoBA",
        category=ImportWarning
    )
    moba_attn_varlen = None
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
    from natten.functional import na2d
except ImportError:
    warnings.warn(
        "NATTEN is not installed. Please install it via `pip install natten",
        category=ImportWarning
    )
    na2d = None
try:
    from native_sparse_attention.ops.parallel import parallel_nsa, parallel_nsa_compression
except ImportError:
    warnings.warn(
        "Native Sparse Attention is not installed. Please check the package installation.",
        category=ImportWarning
    )
    parallel_nsa = None
    parallel_nsa_compression = None

from .utils import _calc_chunks, compress_seq, decompress_seq

try:
    from torch.nn.attention.flex_attention import flex_attention
    from torch.nn.attention.flex_attention import create_block_mask
    flex_attention = torch.compile(flex_attention)
except ImportError:
    warnings.warn(
        "Flex Attention is not installed. Please install it via `pip install torch`",
        category=ImportWarning
    )
    flex_attention = None

WINDOW_SIZE_1D = 256

def sliding_window_1d(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx <= (WINDOW_SIZE_1D // 2)) & (q_idx - kv_idx >= -(WINDOW_SIZE_1D // 2))

def generate_sta_mask_2d(
    H_DIM: int = 16,
    W_DIM: int = 16,
    WINDOW_SIZE_2D_H: int = 12,
    WINDOW_SIZE_2D_W: int = 12,
    TILE_SIZE_2D_H: int = 4,
    TILE_SIZE_2D_W: int = 4,
):
    """
    Generates a sliding tile attention mask for 2D sequence.
    Args:
        H_DIM (int): Height of the input 2D sequence.
        W_DIM (int): Width of the input 2D sequence.
        WINDOW_SIZE_2D_H (int): Height of the sliding window.
        WINDOW_SIZE_2D_W (int): Width of the sliding window.
        TILE_SIZE_2D_H (int): Height of the tile size.
        TILE_SIZE_2D_W (int): Width of the tile size.
    Returns:
        mask (torch.Tensor): Sliding window attention mask for 2D sequence.
    """
    assert H_DIM % TILE_SIZE_2D_H == 0, f"H_DIM {H_DIM} is not divisible by TILE_SIZE_2D_H {TILE_SIZE_2D_H}"
    assert W_DIM % TILE_SIZE_2D_W == 0, f"W_DIM {W_DIM} is not divisible by TILE_SIZE_2D_W {TILE_SIZE_2D_W}"
    assert WINDOW_SIZE_2D_H % TILE_SIZE_2D_H == 0, f"WINDOW_SIZE_2D_H {WINDOW_SIZE_2D_H} is not divisible by TILE_SIZE_2D_H {TILE_SIZE_2D_H}"
    assert WINDOW_SIZE_2D_W % TILE_SIZE_2D_W == 0, f"WINDOW_SIZE_2D_W {WINDOW_SIZE_2D_W} is not divisible by TILE_SIZE_2D_W {TILE_SIZE_2D_W}"

    def sta_mask_mod_2d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        tile_numel = TILE_SIZE_2D_H * TILE_SIZE_2D_W
        tile_idx_q = q_idx // tile_numel
        tile_idx_kv = kv_idx // tile_numel
        h_dim = H_DIM // TILE_SIZE_2D_H
        w_dim = W_DIM // TILE_SIZE_2D_W
        tile_h_q = tile_idx_q // w_dim
        tile_w_q = tile_idx_q % w_dim
        tile_h_kv = tile_idx_kv // w_dim
        tile_w_kv = tile_idx_kv % w_dim

        window_size_h = WINDOW_SIZE_2D_H // TILE_SIZE_2D_H
        window_size_w = WINDOW_SIZE_2D_W // TILE_SIZE_2D_W

        window_size_left_h = window_size_h // 2
        window_size_right_h = window_size_h // 2 + (window_size_h % 2 - 1)
        window_size_left_w = window_size_w // 2
        window_size_right_w = window_size_w // 2 + (window_size_w % 2 - 1)
        
        window_center_h = tile_h_q.clamp(
            window_size_left_h, h_dim - 1 - window_size_right_h
        )
        window_center_w = tile_w_q.clamp(
            window_size_left_w, w_dim - 1 - window_size_right_w
        )

        h_mask = (
            tile_h_kv >= window_center_h - window_size_left_h
        ) & (
            tile_h_kv <= window_center_h + window_size_right_h
        )

        w_mask = (
            tile_w_kv >= window_center_w - window_size_left_w
        ) & (
            tile_w_kv <= window_center_w + window_size_right_w
        )
        
        return h_mask & w_mask
    
    sta_mask_mod_2d.__name__ = f"sta2d_h{H_DIM}_w{W_DIM}_wh{WINDOW_SIZE_2D_H}_ww{WINDOW_SIZE_2D_W}_th{TILE_SIZE_2D_H}_tw{TILE_SIZE_2D_W}"
    return sta_mask_mod_2d

def generate_sta_mask_3d(
    T_DIM: int = 8,
    H_DIM: int = 16,
    W_DIM: int = 16,
    WINDOW_SIZE_3D_T: int = 6,
    WINDOW_SIZE_3D_H: int = 12,
    WINDOW_SIZE_3D_W: int = 12,
    TILE_SIZE_3D_T: int = 2,
    TILE_SIZE_3D_H: int = 4,
    TILE_SIZE_3D_W: int = 4,
):
    """
    Generates a sliding tile attention mask for 3D spatio-temporal sequence.
    Args:
        T_DIM (int): Temporal dimension of the input 3D sequence.
        H_DIM (int): Height of the input 3D sequence.
        W_DIM (int): Width of the input 3D sequence.
        WINDOW_SIZE_3D_T (int): Size of the sliding window along temporal dimension.
        WINDOW_SIZE_3D_H (int): Height of the sliding window in spatial dimensions.
        WINDOW_SIZE_3D_W (int): Width of the sliding window in spatial dimensions.
        TILE_SIZE_3D_T (int): Size of the tile along temporal dimension.
        TILE_SIZE_3D_H (int): Height of the tile size in spatial dimensions.
        TILE_SIZE_3D_W (int): Width of the tile size in spatial dimensions.
    Returns:
        mask (torch.Tensor): Sliding window attention mask for 3D sequence.
    """
    assert T_DIM % TILE_SIZE_3D_T == 0, f"T_DIM {T_DIM} is not divisible by TILE_SIZE_3D_T {TILE_SIZE_3D_T}"
    assert H_DIM % TILE_SIZE_3D_H == 0, f"H_DIM {H_DIM} is not divisible by TILE_SIZE_3D_H {TILE_SIZE_3D_H}"
    assert W_DIM % TILE_SIZE_3D_W == 0, f"W_DIM {W_DIM} is not divisible by TILE_SIZE_3D_W {TILE_SIZE_3D_W}"
    assert WINDOW_SIZE_3D_T % TILE_SIZE_3D_T == 0, f"WINDOW_SIZE_3D_T {WINDOW_SIZE_3D_T} is not divisible by TILE_SIZE_3D_T {TILE_SIZE_3D_T}"
    assert WINDOW_SIZE_3D_H % TILE_SIZE_3D_H == 0, f"WINDOW_SIZE_3D_H {WINDOW_SIZE_3D_H} is not divisible by TILE_SIZE_3D_H {TILE_SIZE_3D_H}"
    assert WINDOW_SIZE_3D_W % TILE_SIZE_3D_W == 0, f"WINDOW_SIZE_3D_W {WINDOW_SIZE_3D_W} is not divisible by TILE_SIZE_3D_W {TILE_SIZE_3D_W}"

    def sta_mask_mod_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        tile_numel = TILE_SIZE_3D_T * TILE_SIZE_3D_H * TILE_SIZE_3D_W
        tile_idx_q = q_idx // tile_numel
        tile_idx_kv = kv_idx // tile_numel
        
        hw_dim = (H_DIM // TILE_SIZE_3D_H) * (W_DIM // TILE_SIZE_3D_W)
        t_dim = T_DIM // TILE_SIZE_3D_T
        h_dim = H_DIM // TILE_SIZE_3D_H
        w_dim = W_DIM // TILE_SIZE_3D_W
        
        tile_t_q = tile_idx_q // hw_dim
        hw_idx_q = tile_idx_q % hw_dim
        tile_h_q = hw_idx_q // w_dim
        tile_w_q = hw_idx_q % w_dim
        
        tile_t_kv = tile_idx_kv // hw_dim
        hw_idx_kv = tile_idx_kv % hw_dim
        tile_h_kv = hw_idx_kv // w_dim
        tile_w_kv = hw_idx_kv % w_dim
        
        window_size_t = WINDOW_SIZE_3D_T // TILE_SIZE_3D_T
        window_size_h = WINDOW_SIZE_3D_H // TILE_SIZE_3D_H
        window_size_w = WINDOW_SIZE_3D_W // TILE_SIZE_3D_W

        window_size_left_t = window_size_t // 2
        window_size_right_t = window_size_t // 2 + (window_size_t % 2 - 1)
        window_size_left_h = window_size_h // 2
        window_size_right_h = window_size_h // 2 + (window_size_h % 2 - 1)
        window_size_left_w = window_size_w // 2
        window_size_right_w = window_size_w // 2 + (window_size_w % 2 - 1)
        
        window_center_t = tile_t_q.clamp(
            window_size_left_t, t_dim - 1 - window_size_right_t
        )
        window_center_h = tile_h_q.clamp(
            window_size_left_h, h_dim - 1 - window_size_right_h
        )
        window_center_w = tile_w_q.clamp(
            window_size_left_w, w_dim - 1 - window_size_right_w
        )

        t_mask = (
            tile_t_kv >= window_center_t - window_size_left_t
        ) & (
            tile_t_kv <= window_center_t + window_size_right_t
        )

        h_mask = (
            tile_h_kv >= window_center_h - window_size_left_h
        ) & (
            tile_h_kv <= window_center_h + window_size_right_h
        )

        w_mask = (
            tile_w_kv >= window_center_w - window_size_left_w
        ) & (
            tile_w_kv <= window_center_w + window_size_right_w
        )
        
        return t_mask & h_mask & w_mask
    
    sta_mask_mod_3d.__name__ = f"sta3d_t{T_DIM}_h{H_DIM}_w{W_DIM}_wt{WINDOW_SIZE_3D_T}_wh{WINDOW_SIZE_3D_H}_ww{WINDOW_SIZE_3D_W}_tt{TILE_SIZE_3D_T}_th{TILE_SIZE_3D_H}_tw{TILE_SIZE_3D_W}"
    return sta_mask_mod_3d


"""
Vanilla Self-Attention
Attention implementation used in hybrid model, adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/attn.py
"""

class FullAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        head_dim: int = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx

        # log
        import logging
        logging.info(f"Using FullAttention")

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


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


class Block1DAttention(nn.Module):
    """
    Block 1D Attention \\
    For 1D sequence, attention calculated within each 1D block
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        block_size: int = 32,
        head_dim: int = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.block_size = block_size

        # log
        import logging
        logging.info(f"Using Block1DAttention with block size {self.block_size}")

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        q = rearrange(self.q_proj(hidden_states), 'b s (h d) -> (b s) h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b s (h d) -> (b s) h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b s (h d) -> (b s) h d', h=self.num_kv_heads)

        # calculate cu_seqlens

        cu_seqlens = torch.arange(
            0, 
            batch_size + 1, 
            dtype=torch.int32, 
            device=hidden_states.device
        ) * q_len

        cu_chunk = _calc_chunks(cu_seqlens, self.block_size)

        if flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        o = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_chunk,
            cu_seqlens_k=cu_chunk,
            max_seqlen_q=q_len,
            max_seqlen_k=q_len,
            causal=False, # use non-causal attention for vision
            window_size=(-1, -1)
        )
        o = o.reshape(batch_size, q_len, self.hidden_size)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None

class Block2DAttention(nn.Module):
    """
    Block 2D Attention \\
    For 2D sequence, attention calculated within each 2D block \\
    Optional feature: shift blocks before attention computation to capture cross-block information
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        block_size_h: int = 32,
        block_size_w: int = 32,
        shift_block: bool = False,  # Whether to shift blocks before attention
        head_dim: int = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w
        self.shift_block = shift_block

        import logging
        logging.info(f"Using Block2DAttention with block size ({self.block_size_h}, {self.block_size_w}) and shift_block={self.shift_block}")
        
        # Calculate shift sizes if shift_block is enabled
        # Find closest multiple of 3 to half of block size
        if self.shift_block:
            self.shift_size_h = 1 + self.layer_idx % (self.block_size_h - 1)
            self.shift_size_w = 1 + self.layer_idx % (self.block_size_w - 1)

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        h_dim: int = None,
        w_dim: int = None, # for custom 2d data size
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)
        
        if h_dim is None:
            h_dim = int(math.sqrt(q_len))
        if w_dim is None:
            w_dim = int(math.sqrt(q_len))

        assert h_dim % self.block_size_h == 0, f"X dim size {h_dim} is not divisible by block size {self.block_size_h}"
        assert w_dim % self.block_size_w == 0, f"Y dim size {w_dim} is not divisible by block size {self.block_size_w}"

        # Apply shifting if enabled
        if self.shift_block:
            # Reshape to 2D spatial dimensions for shift operation
            hidden_states_2d = hidden_states.view(batch_size, h_dim, w_dim, -1)
            
            # Cyclic shift operation
            shifted_hidden_states = torch.roll(
                hidden_states_2d, 
                shifts=(-self.shift_size_h, -self.shift_size_w), 
                dims=(1, 2)
            )
            
            # Convert back to sequence format
            hidden_states = shifted_hidden_states.view(batch_size, q_len, -1)
        
        q = rearrange(self.q_proj(hidden_states), 'b (bnx bsx bny bsy) (h d) -> (b bnx bny) (bsx bsy) h d', bnx=h_dim//self.block_size_h, bny=w_dim//self.block_size_w, bsx=self.block_size_h, bsy=self.block_size_w, h=self.num_heads, d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), 'b (bnx bsx bny bsy) (h d) -> (b bnx bny) (bsx bsy) h d', bnx=h_dim//self.block_size_h, bny=w_dim//self.block_size_w, bsx=self.block_size_h, bsy=self.block_size_w, h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), 'b (bnx bsx bny bsy) (h d) -> (b bnx bny) (bsx bsy) h d', bnx=h_dim//self.block_size_h, bny=w_dim//self.block_size_w, bsx=self.block_size_h, bsy=self.block_size_w, h=self.num_kv_heads, d=self.head_dim)

        if flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # Compute attention using flash attention
        o = flash_attn_func(
            q, k, v,
            causal=False, # use non-causal attention for vision
            window_size=(-1, -1)
        )
        
        # Reshape output back to sequence format
        o = o.reshape(batch_size, q_len, self.hidden_size)
        
        # Reverse shift if shifting was applied
        if self.shift_block:
            o_2d = o.view(batch_size, h_dim, w_dim, -1)
            # Reverse cyclic shift
            o_2d = torch.roll(
                o_2d, 
                shifts=(self.shift_size_h, self.shift_size_w), 
                dims=(1, 2)
            )
            o = o_2d.view(batch_size, q_len, -1)
        
        # Final projection
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None

class Block3DAttention(nn.Module):
    """
    Block 3D Attention \\
    For 3D sequence, attention calculated within each 3D block
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        block_size_h: int = 32,
        block_size_w: int = 32,
        block_size_t: int = 32,
        head_dim: int = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w
        self.block_size_t = block_size_t

        # log
        import logging
        logging.info(f"Using Block3DAttention with block size ({self.block_size_t}, {self.block_size_h}, {self.block_size_w})")

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        h_dim: int = None,
        w_dim: int = None, 
        t_dim: int = None, # for custom 3d data size
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)
        
        if h_dim is None:
            h_dim = int(math.sqrt(q_len))
        if w_dim is None:
            w_dim = int(math.sqrt(q_len))
        if t_dim is None:
            t_dim = int(math.sqrt(q_len))

        assert h_dim % self.block_size_h == 0, f"X dim size {h_dim} is not divisible by block size {self.block_size_h}"
        assert w_dim % self.block_size_w == 0, f"Y dim size {w_dim} is not divisible by block size {self.block_size_w}"
        assert t_dim % self.block_size_t == 0, f"Z dim size {t_dim} is not divisible by block size {self.block_size_t}"

        q = rearrange(self.q_proj(hidden_states), 'b (bnz bsz bnx bsx bny bsy) (h d) -> (b bnz bnx bny) (bsz bsx bsy) h d', bnx=h_dim//self.block_size_h, bny=w_dim//self.block_size_w, bnz=t_dim//self.block_size_t, bsx=self.block_size_h, bsy=self.block_size_w, bsz=self.block_size_t, h=self.num_heads, d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), 'b (bnz bsz bnx bsx bny bsy) (h d) -> (b bnz bnx bny) (bsz bsx bsy) h d', bnx=h_dim//self.block_size_h, bny=w_dim//self.block_size_w, bnz=t_dim//self.block_size_t, bsx=self.block_size_h, bsy=self.block_size_w, bsz=self.block_size_t, h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), 'b (bnz bsz bnx bsx bny bsy) (h d) -> (b bnz bnx bny) (bsz bsx bsy) h d', bnx=h_dim//self.block_size_h, bny=w_dim//self.block_size_w, bnz=t_dim//self.block_size_t, bsx=self.block_size_h, bsy=self.block_size_w, bsz=self.block_size_t, h=self.num_kv_heads, d=self.head_dim)

        if flash_attn_varlen_func is None:
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


class SlidingWindowAttention(nn.Module):

    """
        Sliding Window Attention \\
        Sliding window attention implementation used in hybrid model \\
        Note that this is for 1D sequence. \\
        Use block2d for 2D image and block3d for video.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size: int = 256,  # sliding window size
        head_dim: int = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        backend: str = "flash_attn",
        seq_len: Optional[int] = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.backend = backend
        self.seq_len = seq_len

        if self.backend == "flex_attn":
            assert self.seq_len is not None, "seq_len must be provided for flex_attn"
            global WINDOW_SIZE_1D
            WINDOW_SIZE_1D = self.window_size
            # cache the block mask.
            self.block_mask = create_block_mask(mask_mod=sliding_window_1d, B=None, H=None, Q_LEN=self.seq_len, KV_LEN=self.seq_len, device="cuda")

        # log about backend and window size
        import logging
        logging.info(f"Using {self.backend} backend for sliding window attention with window size {self.window_size}")
        logging.info(f"Note that this is for 1D sequence. Although it can be used for 2D image and 3D video.")

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        if self.backend == "flash_attn":
            q = rearrange(self.q_proj(hidden_states), 'b s (h d) -> b s h d', h=self.num_heads)
            k = rearrange(self.k_proj(hidden_states), 'b s (h d) -> b s h d', h=self.num_kv_heads)
            v = rearrange(self.v_proj(hidden_states), 'b s (h d) -> b s h d', h=self.num_kv_heads)
        elif self.backend == "flex_attn":
            q = rearrange(self.q_proj(hidden_states), 'b s (h d) -> b h s d', h=self.num_heads)
            k = rearrange(self.k_proj(hidden_states), 'b s (h d) -> b h s d', h=self.num_kv_heads)
            v = rearrange(self.v_proj(hidden_states), 'b s (h d) -> b h s d', h=self.num_kv_heads)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # Use Flash Attention with window_size parameter for sliding window attention
        if self.backend == "flash_attn":
            o = flash_attn_func(
                q, k, v,
                causal=False,
                window_size=(self.window_size // 2, self.window_size // 2)  # symmetric window for non-causal attention
            )
        elif self.backend == "flex_attn":
            # change global varibale WINDOW_SIZE to self.window_size
            o = flex_attention(
                q, k, v,
                block_mask=self.block_mask,
            )

        o = o.reshape(batch_size, seq_len, self.hidden_size)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None
    
class SlidingTileAttention2D(nn.Module):
    """
        Sliding Tile Attention \\
        Sliding window attention for 2D used in hybrid model \\
        Note that this is for 2D sequence. \\
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size_h: int = 16,
        window_size_w: int = 16,
        tile_size_h: int = 8,
        tile_size_w: int = 8,
        head_dim: int = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        seq_len: int = 256,
        h_dim: Optional[int] = None,
        w_dim: Optional[int] = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        
        # Window and tile sizes for 2D data
        self.window_size_h = window_size_h
        self.window_size_w = window_size_w
        self.tile_size_h = tile_size_h
        self.tile_size_w = tile_size_w

        # Validate tile size for flex attention requirements
        assert (self.tile_size_h * self.tile_size_w) % 128 == 0, \
            f"tile numel {self.tile_size_h * self.tile_size_w} is not divisible by 128, which is required for flex attention"

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.w_dim = w_dim

        # Calculate dimensions if not provided
        if self.h_dim is None:
            self.h_dim = int(math.sqrt(self.seq_len))
            
        if self.w_dim is None:
            self.w_dim = int(math.sqrt(self.seq_len))

        # Validate sequence length against dimensions
        expected_seq_len = self.h_dim * self.w_dim
        assert self.seq_len == expected_seq_len, f"seq_len {self.seq_len} does not match product of dimensions {expected_seq_len}"
        assert self.seq_len % (self.tile_size_h * self.tile_size_w) == 0, \
            f"seq_len {self.seq_len} is not divisible by (TILE_SIZE_2D_H * TILE_SIZE_2D_W) {self.tile_size_h * self.tile_size_w}"

        # Log configuration
        import logging
        logging.info(f"Using SlidingTileAttention2D with window size ({self.window_size_h}, {self.window_size_w}) "
                     f"and tile size ({self.tile_size_h}, {self.tile_size_w})")

        # Initialize layers
        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Create block mask for 2D sliding tile attention
        sta2d_mask = generate_sta_mask_2d(
            H_DIM=self.h_dim,
            W_DIM=self.w_dim,
            WINDOW_SIZE_2D_H=self.window_size_h,
            WINDOW_SIZE_2D_W=self.window_size_w,
            TILE_SIZE_2D_H=self.tile_size_h,
            TILE_SIZE_2D_W=self.tile_size_w,
        )

        self.block_mask = create_block_mask(
            mask_mod=sta2d_mask, 
            B=None, 
            H=None, 
            Q_LEN=self.seq_len, 
            KV_LEN=self.seq_len, 
            device="cuda", 
            _compile=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        h_dim: int = None,
        w_dim: int = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()

        h_dim = h_dim or self.h_dim
        w_dim = w_dim or self.w_dim

        assert h_dim * w_dim == seq_len, f"Dimensions {h_dim}x{w_dim} don't match sequence length {seq_len}"

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        q = rearrange(
            self.q_proj(hidden_states), 
            'b (nth th ntw tw) (h d) -> b h (nth ntw th tw) d', 
            h=self.num_heads, 
            nth=h_dim // self.tile_size_h, 
            ntw=w_dim // self.tile_size_w, 
            th=self.tile_size_h, 
            tw=self.tile_size_w
        )
        
        k = rearrange(
            self.k_proj(hidden_states), 
            'b (nth th ntw tw) (h d) -> b h (nth ntw th tw) d', 
            h=self.num_kv_heads, 
            nth=h_dim // self.tile_size_h, 
            ntw=w_dim // self.tile_size_w, 
            th=self.tile_size_h, 
            tw=self.tile_size_w
        )
        
        v = rearrange(
            self.v_proj(hidden_states), 
            'b (nth th ntw tw) (h d) -> b h (nth ntw th tw) d', 
            h=self.num_kv_heads, 
            nth=h_dim // self.tile_size_h, 
            ntw=w_dim // self.tile_size_w, 
            th=self.tile_size_h, 
            tw=self.tile_size_w
        )

        if flex_attention is None:
            raise ImportError("Please install Flex Attention via `pip install torch` first")

        o = flex_attention(
            q, k, v,
            block_mask=self.block_mask,
        )
        
        o = rearrange(
            o, 
            'b h (nth ntw th tw) d -> b (nth th ntw tw) (h d)', 
            h=self.num_heads, 
            nth=h_dim // self.tile_size_h, 
            ntw=w_dim // self.tile_size_w, 
            th=self.tile_size_h, 
            tw=self.tile_size_w
        )
        
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None


class SlidingTileAttention3D(nn.Module):
    """
        Sliding Tile Attention for 3D spatio-temporal data \\
        Sliding window attention for video data (TxHxW) used in hybrid model \\
        Extends the 2D STA to handle temporal dimension efficiently
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size_t: int = 6,
        window_size_h: int = 16,
        window_size_w: int = 16,
        tile_size_t: int = 2,
        tile_size_h: int = 8,
        tile_size_w: int = 8,
        head_dim: int = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        seq_len: int = 2048,
        t_dim: Optional[int] = None,
        h_dim: Optional[int] = None,
        w_dim: Optional[int] = None,
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
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        
        # Window and tile sizes for 3D data
        self.window_size_t = window_size_t
        self.window_size_h = window_size_h
        self.window_size_w = window_size_w
        self.tile_size_t = tile_size_t
        self.tile_size_h = tile_size_h
        self.tile_size_w = tile_size_w

        # Validate tile size for flex attention requirements
        assert (self.tile_size_t * self.tile_size_h * self.tile_size_w) % 128 == 0, \
            f"tile numel {self.tile_size_t * self.tile_size_h * self.tile_size_w} is not divisible by 128, which is required for flex attention"

        self.seq_len = seq_len
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        # Calculate dimensions if not provided
        if self.t_dim is None or self.h_dim is None or self.w_dim is None:
            # For 3D data, use default cubic root heuristic if dimensions not specified
            cube_root = round(self.seq_len ** (1/3))
            if self.t_dim is None:
                self.t_dim = cube_root
            if self.h_dim is None:
                self.h_dim = cube_root
            if self.w_dim is None:
                self.w_dim = cube_root

        # Validate sequence length against dimensions
        expected_seq_len = self.t_dim * self.h_dim * self.w_dim
        assert self.seq_len == expected_seq_len, f"seq_len {self.seq_len} does not match product of dimensions {expected_seq_len}"
        assert self.seq_len % (self.tile_size_t * self.tile_size_h * self.tile_size_w) == 0, \
            f"seq_len {self.seq_len} is not divisible by (TILE_SIZE_3D_T * TILE_SIZE_3D_H * TILE_SIZE_3D_W) {self.tile_size_t * self.tile_size_h * self.tile_size_w}"

        # Log configuration
        import logging
        logging.info(f"Using SlidingTileAttention3D with window size ({self.window_size_t}, {self.window_size_h}, {self.window_size_w}) "
                     f"and tile size ({self.tile_size_t}, {self.tile_size_h}, {self.tile_size_w})")

        # Initialize layers
        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        sta3d_mask = generate_sta_mask_3d(
            T_DIM=self.t_dim,
            H_DIM=self.h_dim,
            W_DIM=self.w_dim,
            WINDOW_SIZE_3D_T=self.window_size_t,
            WINDOW_SIZE_3D_H=self.window_size_h,
            WINDOW_SIZE_3D_W=self.window_size_w,
            TILE_SIZE_3D_T=self.tile_size_t,
            TILE_SIZE_3D_H=self.tile_size_h,
            TILE_SIZE_3D_W=self.tile_size_w,
        )

        self.block_mask = create_block_mask(
            mask_mod=sta3d_mask, 
            B=None, 
            H=None, 
            Q_LEN=self.seq_len, 
            KV_LEN=self.seq_len, 
            device="cuda", 
            _compile=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        t_dim: int = None,
        h_dim: int = None,
        w_dim: int = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()

        t_dim = t_dim or self.t_dim
        h_dim = h_dim or self.h_dim
        w_dim = w_dim or self.w_dim

        assert t_dim * h_dim * w_dim == seq_len, f"Dimensions {t_dim}x{h_dim}x{w_dim} don't match sequence length {seq_len}"

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        q = rearrange(
            self.q_proj(hidden_states), 
            'b (ntt tt nth th ntw tw) (h d) -> b h (ntt nth ntw tt th tw) d', 
            h=self.num_heads, 
            ntt=t_dim // self.tile_size_t,
            ntw=w_dim // self.tile_size_w,
            nth=h_dim // self.tile_size_h,
            tt=self.tile_size_t,
            tw=self.tile_size_w,
            th=self.tile_size_h
        )
        
        k = rearrange(
            self.k_proj(hidden_states), 
            'b (ntt tt nth th ntw tw) (h d) -> b h (ntt nth ntw tt th tw) d', 
            h=self.num_kv_heads, 
            ntt=t_dim // self.tile_size_t,
            ntw=w_dim // self.tile_size_w,
            nth=h_dim // self.tile_size_h,
            tt=self.tile_size_t,
            tw=self.tile_size_w,
            th=self.tile_size_h
        )
        
        v = rearrange(
            self.v_proj(hidden_states), 
            'b (ntt tt nth th ntw tw) (h d) -> b h (ntt nth ntw tt th tw) d', 
            h=self.num_kv_heads, 
            ntt=t_dim // self.tile_size_t,
            ntw=w_dim // self.tile_size_w,
            nth=h_dim // self.tile_size_h,
            tt=self.tile_size_t,
            tw=self.tile_size_w,
            th=self.tile_size_h
        )

        if flex_attention is None:
            raise ImportError("Please install Flex Attention via `pip install torch` first")

        o = flex_attention(
            q, k, v,
            block_mask=self.block_mask,
        )
        
        o = rearrange(
            o, 
            'b h (ntt nth ntw tt th tw) d -> b (ntt tt nth th ntw tw) (h d)', 
            h=self.num_heads, 
            ntt=t_dim // self.tile_size_t,
            ntw=w_dim // self.tile_size_w,
            nth=h_dim // self.tile_size_h,
            tt=self.tile_size_t,
            tw=self.tile_size_w,
            th=self.tile_size_h
        )
        
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None

class Neighborhood2DAttention(nn.Module):
    """
    Basically its a 2D version of sliding window attention. \\
    This one uses NATTEN backend.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        block_size_x: int = 32,
        block_size_y: int = 32,
        head_dim: int = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ):
        super().__init__()

        import logging
        logging.warning("Using NATTEN for 2D data, this therotically is linear time and you may expect it to be MUCH faster than full attention, but actually nope. \\")
        logging.warning("See details in Table 1 from https://hao-ai-lab.github.io/blogs/sta/")

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.block_size_x = block_size_x
        self.block_size_y = block_size_y

        # log
        import logging
        logging.info(f"Using Neighborhood2DAttention with block size ({self.block_size_x}, {self.block_size_y})")

        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        x_dim: int = None,
        y_dim: int = None, # for custom 2d data size
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)
        
        if x_dim is None:
            x_dim = int(math.sqrt(q_len))
        if y_dim is None:
            y_dim = int(math.sqrt(q_len))

        assert x_dim % self.block_size_x == 0, f"X dim size {x_dim} is not divisible by block size {self.block_size_x}"
        assert y_dim % self.block_size_y == 0, f"Y dim size {y_dim} is not divisible by block size {self.block_size_y}"

        q = rearrange(self.q_proj(hidden_states), 'b (x y) (h d) -> b x y h d', x=x_dim, y=y_dim, h=self.num_heads, d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), 'b (x y) (h d) -> b x y h d', x=x_dim, y=y_dim, h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), 'b (x y) (h d) -> b x y h d', x=x_dim, y=y_dim, h=self.num_kv_heads, d=self.head_dim)

        if na2d is None:
            raise ImportError("Please install NATTEN via `pip install natten` first")

        o = na2d(q, k, v, kernel_size=(self.block_size_x, self.block_size_y))
        o = o.reshape(batch_size, q_len, self.hidden_size)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None


"""
Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
NativeSparseAttention simplified implementation, adapted from https://github.com/fla-org/native-sparse-attention
# TODO: currently the kernel of NSA is causal, need to make it non-causal and work for vision
"""

class NativeSparseAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 64,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = None,
        qkv_bias: bool = False,
        block_size: Optional[int] = 64,
        block_counts: Optional[Union[torch.LongTensor, int]] = 16,
        window_size: Optional[int] = 512,
        layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads

        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias

        self.block_size = block_size
        self.block_counts = block_counts
        self.window_size = window_size
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * 3, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, seq_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

        seqlen_offset, max_seqlen = 0, seq_len

        o = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_size=self.block_size,
            block_counts=self.block_counts,
            window_size=self.window_size,
            head_first=False
        )
        o = o.reshape(batch_size, seq_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None

"""
MoBA: Mixture of Block Attention for Long-Context LLMs
Simplified implementation for vision model, adapted from https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_efficient.py
# TODO: currently the kernel of MoBA is causal, need to make it non-causal and work for vision
"""

class MoBA(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        head_dim: int = None,
        block_size: int = 64,
        topk: int = 3,
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
        self.hidden_size = hidden_size
        
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = head_dim
            
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        
        self.block_size = block_size  # Block size for MoBA
        self.topk = topk  # Number of blocks to select for attention

        # Layernorm for normalization-first architecture
        if norm_first:
            self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)
            
        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through VisionMoBA attention layer.
        Uses the moba_attn_varlen function for efficient sparse attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            output_attentions: Whether to output attention weights (not implemented for MoBA)
            
        Returns:
            output: Output tensor after attention
            attentions: None (not implemented for MoBA)
            past_key_values: None (stateless implementation)
        """
        batch_size, seq_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        q = rearrange(self.q_proj(hidden_states), 'b s (h d) -> (b s) h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b s (h d) -> (b s) h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b s (h d) -> (b s) h d', h=self.num_kv_heads)
        
        # If grouped query attention is used, repeat k and v to match num_heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # For vision models, all sequences are the same length
        cu_seqlens = torch.arange(
            0, 
            batch_size + 1, 
            dtype=torch.int32, 
            device=hidden_states.device
        ) * seq_len
        
        o = moba_attn_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            moba_chunk_size=self.block_size,
            moba_topk=self.topk
        )
        
        o = o.reshape(batch_size, seq_len, self.hidden_size)
        o = self.o_proj(o)

        attentions = None
        
        return o, attentions, None


ATTN_LISTS = ["full_attn", "moba", "nsa", "block1d_attn", "block2d_attn", "sw_attn", "sta2d_attn", "na2d_attn"]
def get_attn(config, layer_idx):
    """
    This is for full/local/sparse attention, not linear attention
    """
    attn_type = config.attn_type
    assert attn_type in ATTN_LISTS, f"Attention type must be one of {ATTN_LISTS}"

    if attn_type == "full_attn":
        return FullAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            layer_idx=layer_idx
        )
    elif attn_type == "moba":
        return MoBA(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            block_size=config.attn["block_size"],
            topk=config.attn["topk"],
            layer_idx=layer_idx
        )
    elif attn_type == "nsa":
        return NativeSparseAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            block_size=config.attn['block_size'],
            block_counts=config.attn['block_counts'],
            window_size=config.attn['window_size'],
            layer_idx=layer_idx
        )
    elif attn_type == "block1d_attn":
        return Block1DAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            block_size=config.attn['block_size'],
            layer_idx=layer_idx
        )
    elif attn_type == "block2d_attn":
        return Block2DAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            block_size_h=config.attn['block_size_h'],
            block_size_w=config.attn['block_size_w'],
            layer_idx=layer_idx
        )
    elif attn_type == "sw_attn":
        return SlidingWindowAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            window_size=config.attn['window_size'],
            layer_idx=layer_idx
        )
    elif attn_type == "sta2d_attn":
        return SlidingTileAttention2D(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            window_size_h=config.attn['window_size_h'],
            window_size_w=config.attn['window_size_w'],
            tile_size_h=config.attn['tile_size_h'],
            tile_size_w=config.attn['tile_size_w'],
            seq_len=config.attn['seq_len'],
            layer_idx=layer_idx
        )
    elif attn_type == "na2d_attn":
        return Neighborhood2DAttention(
            hidden_size=config.hidden_size,
            num_heads=config.attn['num_heads'],
            num_kv_heads=config.attn['num_kv_heads'],
            block_size_x=config.attn['block_size_x'],
            block_size_y=config.attn['block_size_y'],
            layer_idx=layer_idx
        )
    else:
        raise ValueError(f"Attention type {attn_type} is not supported")
