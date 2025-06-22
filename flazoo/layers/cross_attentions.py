# -*- coding: utf-8 -*-

from diffusers.models.attention_processor import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, Unpack
from fla.modules import ShortConvolution, RMSNorm, FusedRMSNormGated
from fla.models.utils import Cache
from fla.ops import (
    fused_recurrent_delta_rule,
    chunk_delta_rule
)
from flazoo.ops import (
    generate_sta_mask_3d,
    sta_3d_with_text_func
)
import warnings

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

class DeltaNetCrossAttentionHF(Attention):
    """
    A simplified DeltaNet layer handling cross attention API compatable with HuggingFace's Diffusers library.
    """
    def __init__(
        self,
        fla_config: Dict = None,
        layer_idx: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.layer_idx = layer_idx
        
        # some specific parameters for the DeltaNet layer
        self.train_scan_type = fla_config.get('train_scan_type', 'uni-scan')
        self.test_scan_type = fla_config.get('test_scan_type', 'uni-scan')
        self.mode = fla_config.get('mode', 'chunk')
        self.use_beta = fla_config.get('use_beta', True)
        self.use_short_conv = fla_config.get('use_short_conv', True)
        self.conv_size = fla_config.get('conv_size', 4)
        self.conv_bias = fla_config.get('conv_bias', False)
        self.qk_activation = fla_config.get('qk_activation', 'silu')
        self.qk_norm = fla_config.get('qk_norm', 'l2')
        self.use_gate = fla_config.get('use_gate', False)
        self.norm_eps = fla_config.get('norm_eps', 1e-6)
        self.allow_neg_eigval = fla_config.get('allow_neg_eigval', False)
        self.head_v_dim = self.query_dim // self.heads
        
        if self.use_beta:
            self.b_proj = nn.Linear(self.query_dim, self.heads, bias=False)
        
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.inner_dim,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                activation='silu' if self.qk_activation == 'silu' else None
            )
            
            self.k_conv1d = ShortConvolution(
                hidden_size=self.inner_dim,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                activation='silu' if self.qk_activation == 'silu' else None
            )
            
            self.v_conv1d = ShortConvolution(
                hidden_size=self.inner_dim,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                activation='silu'
            )
        
        if self.use_gate:
            self.g_proj = nn.Linear(self.query_dim, self.cross_attention_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=self.norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=self.norm_eps)
        
    def op_forward_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None, # used to calculate extra params
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = q.shape

        last_state = None

        cu_seqlens = kwargs.get('cu_seqlens', None)
        mode = 'fused_recurrent' if q_len <= 64 else self.mode

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=q,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=k,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=v,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(v)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.heads), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', h=self.heads) # TODO: support different kv heads in the future
        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError
            
        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')

            
    
class SlidingTileCrossAttentionHF3D(Attention):
    """
    A simplified STA3D layer handling cross attention API compatable with HuggingFace's Diffusers library.
    """
    def __init__(
        self,
        fla_config: Dict = None,
        layer_idx: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        assert fla_config["attn"] is not None, "fla_config.attn must be provided for STA3DCrossAttentionHF."
        attn_config = fla_config["attn"]

        self.layer_idx = layer_idx
        self.window_size_t = attn_config.get('window_size_t', 24)
        self.window_size_h = attn_config.get('window_size_h', 24)
        self.window_size_w = attn_config.get('window_size_w', 24)
        self.tile_size_t = attn_config.get('tile_size_t', 8)
        self.tile_size_h = attn_config.get('tile_size_h', 8)
        self.tile_size_w = attn_config.get('tile_size_w', 8)
        self.t_dim = attn_config.get('t_dim', 32)
        self.h_dim = attn_config.get('h_dim', 32)
        self.w_dim = attn_config.get('w_dim', 32)
        self.text_seq_len = attn_config.get('text_seq_len', 512)

        self.vision_seq_len = self.t_dim * self.h_dim * self.w_dim

        self.seq_len = self.vision_seq_len + self.text_seq_len

        import os

        compile = os.environ.get('COMPILE_BLOCK_MASK', 'False').lower() in ('true', '1', 'yes')
        warnings.warn(
            f"Using compile={compile} for block mask generation. "
            "Set COMPILE_BLOCK_MASK environment variable to 'True' to enable compilation."
        )

        self.block_mask = generate_sta_mask_3d(
            canvas_thw=(self.t_dim, self.h_dim, self.w_dim),
            kernel_thw=(self.window_size_t, self.window_size_h, self.window_size_w),
            tile_thw=(self.tile_size_t, self.tile_size_h, self.tile_size_w),    
            total_seq_len=self.seq_len,
            text_seq_len=self.text_seq_len,
            compile=compile,
        )
    
    def op_forward_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None, # used to calculate extra params
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ):
        # Note that q, k, v here maybe includes both text and vision tokens.
        # we assume q, k, v is of shape [B, L, D]

        return sta_3d_with_text_func(
            q=q,
            k=k,
            v=v,
            t_dim= self.t_dim,
            h_dim= self.h_dim,
            w_dim= self.w_dim,
            tile_size_t=self.tile_size_t,
            tile_size_h=self.tile_size_h,
            tile_size_w=self.tile_size_w,
            block_mask=self.block_mask,
            text_seq_len=self.text_seq_len,
            num_heads=self.heads,
            num_kv_heads= self.heads, # TODO: support different kv heads in the future
        )
        
        return o