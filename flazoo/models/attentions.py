# -*- coding: utf-8 -*-

import triton
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union


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
    from native_sparse_attention.ops.parallel import parallel_nsa, parallel_nsa_compression
except ImportError:
    warnings.warn(
        "Native Sparse Attention is not installed. Please check the package installation.",
        category=ImportWarning
    )
    parallel_nsa = None
    parallel_nsa_compression = None

from .utils import _calc_chunks, compress_seq, decompress_seq

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
    For 2D sequence, attention calculated within each 2D block
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

        q = rearrange(self.q_proj(hidden_states), 'b (bnx bsx bny bsy) (h d) -> (b bnx bny) (bsx bsy) h d', bnx=x_dim//self.block_size_x, bny=y_dim//self.block_size_y, bsx=self.block_size_x, bsy=self.block_size_y, h=self.num_heads, d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), 'b (bnx bsx bny bsy) (h d) -> (b bnx bny) (bsx bsy) h d', bnx=x_dim//self.block_size_x, bny=y_dim//self.block_size_y, bsx=self.block_size_x, bsy=self.block_size_y, h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), 'b (bnx bsx bny bsy) (h d) -> (b bnx bny) (bsx bsy) h d', bnx=x_dim//self.block_size_x, bny=y_dim//self.block_size_y, bsx=self.block_size_x, bsy=self.block_size_y, h=self.num_kv_heads, d=self.head_dim)

        if flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        o = flash_attn_varlen_func(
            q, k, v,
            causal=False, # use non-causal attention for vision
            window_size=(-1, -1)
        )
        o = o.reshape(batch_size, q_len, self.hidden_size)
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
        block_size_x: int = 32,
        block_size_y: int = 32,
        block_size_z: int = 32,
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
        self.block_size_x = block_size_x
        self.block_size_y = block_size_y
        self.block_size_z = block_size_z

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
        y_dim: int = None, 
        z_dim: int = None, # for custom 3d data size
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        if self.norm_first:
            hidden_states = self.norm(hidden_states)
        
        if x_dim is None:
            x_dim = int(math.sqrt(q_len))
        if y_dim is None:
            y_dim = int(math.sqrt(q_len))
        if z_dim is None:
            z_dim = int(math.sqrt(q_len))

        assert x_dim % self.block_size_x == 0, f"X dim size {x_dim} is not divisible by block size {self.block_size_x}"
        assert y_dim % self.block_size_y == 0, f"Y dim size {y_dim} is not divisible by block size {self.block_size_y}"
        assert z_dim % self.block_size_z == 0, f"Z dim size {z_dim} is not divisible by block size {self.block_size_z}"

        q = rearrange(self.q_proj(hidden_states), 'b (bnx bsx bny bsy bnz bsz) (h d) -> (b bnx bny bnz) (bsx bsy bsz) h d', bnx=x_dim//self.block_size_x, bny=y_dim//self.block_size_y, bnz=z_dim//self.block_size_z, bsx=self.block_size_x, bsy=self.block_size_y, bsz=self.block_size_z, h=self.num_heads, d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), 'b (bnx bsx bny bsy bnz bsz) (h d) -> (b bnx bny bnz) (bsx bsy bsz) h d', bnx=x_dim//self.block_size_x, bny=y_dim//self.block_size_y, bnz=z_dim//self.block_size_z, bsx=self.block_size_x, bsy=self.block_size_y, bsz=self.block_size_z, h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), 'b (bnx bsx bny bsy bnz bsz) (h d) -> (b bnx bny bnz) (bsx bsy bsz) h d', bnx=x_dim//self.block_size_x, bny=y_dim//self.block_size_y, bnz=z_dim//self.block_size_z, bsx=self.block_size_x, bsy=self.block_size_y, bsz=self.block_size_z, h=self.num_kv_heads, d=self.head_dim)

        if flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        o = flash_attn_varlen_func(
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

        q = rearrange(self.q_proj(hidden_states), 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b s (h d) -> b s h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b s (h d) -> b s h d', h=self.num_kv_heads)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # Use Flash Attention with window_size parameter for sliding window attention
        o = flash_attn_func(
            q, k, v,
            causal=False,
            window_size=(self.window_size // 2, self.window_size // 2)  # symmetric window for non-causal attention
        )

        o = o.reshape(batch_size, seq_len, self.hidden_size)
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


ATTN_LISTS = ["full_attn", "moba", "nsa", "block1d_attn", "block2d_attn", "sw_attn"]
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
            block_size_x=config.attn['block_size_x'],
            block_size_y=config.attn['block_size_y'],
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
    else:
        raise ValueError(f"Attention type {attn_type} is not supported")
