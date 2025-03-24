# -*- coding: utf-8 -*-

import triton
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import TYPE_CHECKING, Optional, Tuple, Union
from moba.moba_efficient import moba_attn_varlen
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


"""
Attention implementation used in hybrid model, adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/attn.py
"""

class VisionAttention(nn.Module):

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

"""
NativeSparseAttention simplified implementation, adapted from https://github.com/fla-org/native-sparse-attention
"""

class VisionNativeSparseAttention(nn.Module):

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
"""

class VisionMoBA(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        head_dim: int = None,
        block_size: int = 64,
        topk: int = 8,
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

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        
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
