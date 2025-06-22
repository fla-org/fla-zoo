# -*- coding: utf-8 -*-

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature, BlockMask, create_block_mask, flex_attention
from einops import rearrange
from typing import Tuple
from torch import IntTensor, BoolTensor

flex_attention = torch.compile(flex_attention)

def generate_sta_mask_mod_2d(
    canvas_hw: Tuple[int, int],
    kernel_hw: Tuple[int, int],
    tile_hw: Tuple[int, int],
    text_seq_len: int = 0,
) -> _mask_mod_signature:
    """Generates a 2D STA mask mod with a given kernel size.

    Args:
        canvas_hw (Tuple[int, int]): The shape of the canvas (height, width).
        kernel_hw (Tuple[int, int]): The shape of the kernel (height, width).
        tile_hw (Tuple[int, int]): The shape of the tile (height, width).
        text_seq_len (int): The length of the text sequence for masking.
    """
    canvas_h, canvas_w = canvas_hw
    kernel_h, kernel_w = kernel_hw
    tile_h, tile_w = tile_hw
    tile_numel = tile_h * tile_w
    assert canvas_h % tile_h == 0, (
        f"Canvas height {canvas_h} is not divisible by tile height {tile_h}"
    )
    assert canvas_w % tile_w == 0, (
        f"Canvas width {canvas_w} is not divisible by tile width {tile_w}"
    )
    assert kernel_h % tile_h == 0, (
        f"Kernel height {kernel_h} is not divisible by tile height {tile_h}"
    )
    assert kernel_w % tile_w == 0, (
        f"Kernel width {kernel_w} is not divisible by tile width {tile_w}"
    )
    canvas_tile_h, canvas_tile_w = canvas_h // tile_h, canvas_w // tile_w
    kernel_tile_h, kernel_tile_w = kernel_h // tile_h, kernel_w // tile_w
    vision_seq_len = canvas_h * canvas_w

    def get_h_w_idx_tiled(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        tile_id = idx // tile_numel
        tile_h_idx = tile_id // canvas_tile_w
        tile_w_idx = tile_id % canvas_tile_w
        return tile_h_idx, tile_w_idx

    def get_border(kernel_size: IntTensor) -> Tuple[IntTensor, IntTensor]:
        left_border = kernel_size // 2
        right_border = kernel_size // 2 + (kernel_size % 2 - 1)
        return left_border, right_border

    def sta_mask_mod_2d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_tile_h, q_tile_w = get_h_w_idx_tiled(q_idx)
        kv_tile_h, kv_tile_w = get_h_w_idx_tiled(kv_idx)
        left_border_h, right_border_h = get_border(kernel_tile_h)
        left_border_w, right_border_w = get_border(kernel_tile_w)
        kernel_center_h = q_tile_h.clamp(
            left_border_h, (canvas_tile_h - 1) - right_border_h
        )
        kernel_center_w = q_tile_w.clamp(
            left_border_w, (canvas_tile_w - 1) - right_border_w
        )
        h_mask = (kv_tile_h >= kernel_center_h - left_border_h) & (
            kv_tile_h <= kernel_center_h + right_border_h
        )
        w_mask = (kv_tile_w >= kernel_center_w - left_border_w) & (
            kv_tile_w <= kernel_center_w + right_border_w
        )
        vision_mask = (q_idx < vision_seq_len) & (kv_idx < vision_seq_len)
        vision_to_text_mask = (
            (q_idx < vision_seq_len)
            & (kv_idx >= vision_seq_len)
            & (kv_idx < vision_seq_len + text_seq_len)
        )
        text_to_all_mask = (q_idx >= vision_seq_len) & (
            kv_idx < vision_seq_len + text_seq_len
        )
        return (vision_mask & h_mask & w_mask) | vision_to_text_mask | text_to_all_mask

    sta_mask_mod_2d.__name__ = (
        f"sta_2d_c{canvas_h}x{canvas_w}_k{kernel_h}x{kernel_w}_t{tile_h}x{tile_w}"
    )
    return sta_mask_mod_2d


def generate_sta_mask_mod_3d(
    canvas_thw: Tuple[int, int, int],
    kernel_thw: Tuple[int, int, int],
    tile_thw: Tuple[int, int, int],
    text_seq_len: int = 0,
) -> _mask_mod_signature:
    """Generates a 3D STA mask mod with a given kernel size.

    Args:
        canvas_twh (Tuple[int, int, int]): The shape of the canvas (time, height, width).
        kernel_twh (Tuple[int, int, int]): The shape of the kernel (time, height, width).
        tile_twh (Tuple[int, int, int]): The shape of the tile (time, height, width).
        text_seq_len (int): The length of the text sequence for masking.
    """
    canvas_t, canvas_h, canvas_w = canvas_thw
    kernel_t, kernel_h, kernel_w = kernel_thw
    tile_t, tile_h, tile_w = tile_thw
    tile_numel = tile_t * tile_h * tile_w
    assert canvas_t % tile_t == 0, (
        f"Canvas time {canvas_t} is not divisible by tile time {tile_t}"
    )
    assert canvas_h % tile_h == 0, (
        f"Canvas height {canvas_h} is not divisible by tile height {tile_h}"
    )
    assert canvas_w % tile_w == 0, (
        f"Canvas width {canvas_w} is not divisible by tile width {tile_w}"
    )
    assert kernel_t % tile_t == 0, (
        f"Kernel time {kernel_t} is not divisible by tile time {tile_t}"
    )
    assert kernel_h % tile_h == 0, (
        f"Kernel height {kernel_h} is not divisible by tile height {tile_h}"
    )
    assert kernel_w % tile_w == 0, (
        f"Kernel width {kernel_w} is not divisible by tile width {tile_w}"
    )
    canvas_tile_t, canvas_tile_h, canvas_tile_w = (
        canvas_t // tile_t,
        canvas_h // tile_h,
        canvas_w // tile_w,
    )
    kernel_tile_t, kernel_tile_h, kernel_tile_w = (
        kernel_t // tile_t,
        kernel_h // tile_h,
        kernel_w // tile_w,
    )
    vision_seq_len = canvas_t * canvas_h * canvas_w

    def get_t_h_w_idx_tiled(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        tile_id = idx // tile_numel
        tile_t_idx = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h_idx = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w_idx = tile_id % canvas_tile_w
        return tile_t_idx, tile_h_idx, tile_w_idx

    def get_border(kernel_size: IntTensor) -> Tuple[IntTensor, IntTensor]:
        left_border = kernel_size // 2
        right_border = kernel_size // 2 + (kernel_size % 2 - 1)
        return left_border, right_border

    def sta_mask_mod_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_tile_t, q_tile_h, q_tile_w = get_t_h_w_idx_tiled(q_idx)
        kv_tile_t, kv_tile_h, kv_tile_w = get_t_h_w_idx_tiled(kv_idx)
        left_border_t, right_border_t = get_border(kernel_tile_t)
        left_border_h, right_border_h = get_border(kernel_tile_h)
        left_border_w, right_border_w = get_border(kernel_tile_w)
        kernel_center_t = q_tile_t.clamp(
            left_border_t, (canvas_tile_t - 1) - right_border_t
        )
        kernel_center_h = q_tile_h.clamp(
            left_border_h, (canvas_tile_h - 1) - right_border_h
        )
        kernel_center_w = q_tile_w.clamp(
            left_border_w, (canvas_tile_w - 1) - right_border_w
        )
        t_mask = (kv_tile_t >= kernel_center_t - left_border_t) & (
            kv_tile_t <= kernel_center_t + right_border_t
        )
        h_mask = (kv_tile_h >= kernel_center_h - left_border_h) & (
            kv_tile_h <= kernel_center_h + right_border_h
        )
        w_mask = (kv_tile_w >= kernel_center_w - left_border_w) & (
            kv_tile_w <= kernel_center_w + right_border_w
        )
        vision_mask = (q_idx < vision_seq_len) & (kv_idx < vision_seq_len)
        vision_to_text_mask = (
            (q_idx < vision_seq_len)
            & (kv_idx >= vision_seq_len)
            & (kv_idx < vision_seq_len + text_seq_len)
        )
        text_to_all_mask = (q_idx >= vision_seq_len) & (
            kv_idx < vision_seq_len + text_seq_len
        )
        return (
            (vision_mask & t_mask & w_mask & h_mask)
            | vision_to_text_mask
            | text_to_all_mask
        )

    sta_mask_mod_3d.__name__ = f"sta_3d_c{canvas_t}x{canvas_h}x{canvas_w}_k{kernel_t}x{kernel_h}x{kernel_w}_t{tile_t}x{tile_h}x{tile_w}"
    return sta_mask_mod_3d

def generate_sta_mask_2d(
    canvas_hw: Tuple[int, int],
    kernel_hw: Tuple[int, int],
    tile_hw: Tuple[int, int],
    text_seq_len: int = 0,
    total_seq_len: int = None,
    compile: bool = False,

) -> BlockMask:
    
    if total_seq_len is None:
        total_seq_len = canvas_hw[0] * canvas_hw[1] + text_seq_len
    
    sta2d_mask_mod = generate_sta_mask_mod_2d(
        canvas_hw=canvas_hw,
        kernel_hw=kernel_hw,
        tile_hw=tile_hw,
        text_seq_len=text_seq_len,
    )

    block_mask = create_block_mask(
        mask_mod=sta2d_mask_mod,
        B=None,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
        _compile=compile
    )

    return block_mask

def generate_sta_mask_3d(
    canvas_thw: Tuple[int, int, int],
    kernel_thw: Tuple[int, int, int],
    tile_thw: Tuple[int, int, int],
    text_seq_len: int = 0,
    total_seq_len: int = None,
    compile: bool = False,
) -> BlockMask:
    
    if total_seq_len is None:
        total_seq_len = canvas_thw[0] * canvas_thw[1] * canvas_thw[2] + text_seq_len
    
    sta3d_mask_mod = generate_sta_mask_mod_3d(
        canvas_thw=canvas_thw,
        kernel_thw=kernel_thw,
        tile_thw=tile_thw,
        text_seq_len=text_seq_len,
    )

    block_mask = create_block_mask(
        mask_mod=sta3d_mask_mod,
        B=None,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
        _compile=compile
    )

    return block_mask


def sta_2d_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h_dim: IntTensor,
    w_dim: IntTensor,
    tile_size_h: IntTensor,
    tile_size_w: IntTensor,
    block_mask: BlockMask,
    num_heads: IntTensor,
    num_kv_heads: IntTensor = None,
) -> torch.Tensor:
    """Forward pass for 2D STA
    Args:
        q (torch.Tensor): Query tensor of shape (B, L, D).
        k (torch.Tensor): Key tensor of shape (B, L, D).
        v (torch.Tensor): Value tensor of shape (B, L, D).
        h_dim (IntTensor): Height dimension.
        w_dim (IntTensor): Width dimension.
        tile_size_h (IntTensor): Height tile size.
        tile_size_w (IntTensor): Width tile size.
        block_mask (BlockMask): Block mask for Flex Attention.
    """

    q = rearrange(
        q,
        "b (nth th ntw tw) (h d) -> b h (nth ntw th tw) d",
        h=num_heads,
        nth=h_dim // tile_size_h,
        ntw=w_dim // tile_size_w,
        th=tile_size_h,
        tw=tile_size_w,
    )

    k = rearrange(
        k,
        "b (nth th ntw tw) (h d) -> b h (nth ntw th tw) d",
        h=num_kv_heads,
        nth=h_dim // tile_size_h,
        ntw=w_dim // tile_size_w,
        th=tile_size_h,
        tw=tile_size_w,
    )

    v = rearrange(
        v,
        "b (nth th ntw tw) (h d) -> b h (nth ntw th tw) d",
        h=num_kv_heads,
        nth=h_dim // tile_size_h,
        ntw=w_dim // tile_size_w,
        th=tile_size_h,
        tw=tile_size_w,
    )

    if flex_attention is None:
        raise ImportError(
            "Please install Flex Attention via `pip install torch` first"
        )

    o = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
    )

    o = rearrange(
        o,
        "b h (nth ntw th tw) d -> b (nth th ntw tw) (h d)",
        h=num_heads,
        nth=h_dim // tile_size_h,
        ntw=w_dim // tile_size_w,
        th=tile_size_h,
        tw=tile_size_w,
    )

    return o

def sta_3d_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t_dim: IntTensor,
    h_dim: IntTensor,
    w_dim: IntTensor,
    tile_size_t: IntTensor,
    tile_size_h: IntTensor,
    tile_size_w: IntTensor,
    block_mask: BlockMask,
    num_heads: IntTensor,
    num_kv_heads: IntTensor = None,
) -> torch.Tensor:
    """Forward pass for 3D STA
    Args:
        q (torch.Tensor): Query tensor of shape (B, L, D).
        k (torch.Tensor): Key tensor of shape (B, L, D).
        v (torch.Tensor): Value tensor of shape (B, L, D).
        t_dim (IntTensor): Time dimension.
        h_dim (IntTensor): Height dimension.
        w_dim (IntTensor): Width dimension.
        tile_size_t (IntTensor): Time tile size.
        tile_size_h (IntTensor): Height tile size.
        tile_size_w (IntTensor): Width tile size.
        block_mask (BlockMask): Block mask for Flex Attention.
        num_heads (IntTensor): Number of heads for query.
        num_kv_heads (IntTensor): Number of heads for key and value.
    """

    def tile(x: torch.Tensor, num_of_heads: IntTensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (ntt tt nth th ntw tw) (h d) -> b h (ntt nth ntw tt th tw) d",
            h=num_of_heads,
            ntt=t_dim // tile_size_t,
            ntw=w_dim // tile_size_w,
            nth=h_dim // tile_size_h,
            tt=tile_size_t,
            tw=tile_size_w,
            th=tile_size_h,
        )
    
    def untile(x: torch.Tensor, num_of_heads: IntTensor) -> torch.Tensor:
        return rearrange(
            x,
            "b h (ntt nth ntw tt th tw) d -> b (ntt tt nth th ntw tw) (h d)",
            h=num_of_heads,
            ntt=t_dim // tile_size_t,
            ntw=w_dim // tile_size_w,
            nth=h_dim // tile_size_h,
            tt=tile_size_t,
            tw=tile_size_w,
            th=tile_size_h,
        )
    

    q = tile(q, num_heads)
    k = tile(k, num_kv_heads)
    v = tile(v, num_kv_heads)

    if flex_attention is None:
        raise ImportError(
            "Please install Flex Attention via `pip install torch` first"
        )

    o = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
    )

    o = untile(o, num_heads)

    return o

def sta_3d_with_text_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t_dim: IntTensor,
    h_dim: IntTensor,
    w_dim: IntTensor,
    tile_size_t: IntTensor,
    tile_size_h: IntTensor,
    tile_size_w: IntTensor,
    block_mask: BlockMask,
    text_seq_len: IntTensor,
    num_heads: IntTensor,
    num_kv_heads: IntTensor = None,
) -> torch.Tensor:
    """Forward pass for 3D STA with text.
    Args:
        q (torch.Tensor): Query tensor of shape (B, L, D).
        k (torch.Tensor): Key tensor of shape (B, L, D).
        v (torch.Tensor): Value tensor of shape (B, L, D).
        t_dim (IntTensor): Time dimension.
        h_dim (IntTensor): Height dimension.
        w_dim (IntTensor): Width dimension.
        tile_size_t (IntTensor): Time tile size.
        tile_size_h (IntTensor): Height tile size.
        tile_size_w (IntTensor): Width tile size.
        block_mask (BlockMask): Block mask for Flex Attention.
        text_seq_len (IntTensor): Length of the text sequence.
        num_heads (IntTensor): Number of heads for query.
        num_kv_heads (IntTensor): Number of heads for key and value.
    """

    def split_heads(x: torch.Tensor, num_of_heads: IntTensor) -> torch.Tensor:
        return rearrange(
            x,
            "b L (h d) -> b h L d",
            h=num_of_heads,
        )

    def merge_heads(x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b h L d -> b L (h d)",
        )

    def tile(x: torch.Tensor, num_of_heads: IntTensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (ntt tt nth th ntw tw) (h d) -> b h (ntt nth ntw tt th tw) d",
            h=num_of_heads,
            ntt=t_dim // tile_size_t,
            ntw=w_dim // tile_size_w,
            nth=h_dim // tile_size_h,
            tt=tile_size_t,
            tw=tile_size_w,
            th=tile_size_h,
        )
    
    def untile(x: torch.Tensor, num_of_heads: IntTensor) -> torch.Tensor:
        return rearrange(
            x,
            "b h (ntt nth ntw tt th tw) d -> b (ntt tt nth th ntw tw) (h d)",
            h=num_of_heads,
            ntt=t_dim // tile_size_t,
            ntw=w_dim // tile_size_w,
            nth=h_dim // tile_size_h,
            tt=tile_size_t,
            tw=tile_size_w,
            th=tile_size_h,
        )
    
    vision_seq_len = q.shape[1] - text_seq_len

    q = torch.concat((tile(q[:, :vision_seq_len, :], num_heads), split_heads(q[:, vision_seq_len:, :], num_heads)), dim=2)
    k = torch.concat((tile(k[:, :vision_seq_len, :], num_kv_heads), split_heads(k[:, vision_seq_len:, :], num_kv_heads)), dim=2)
    v = torch.concat((tile(v[:, :vision_seq_len, :], num_kv_heads), split_heads(v[:, vision_seq_len:, :], num_kv_heads)), dim=2)

    if flex_attention is None:
        raise ImportError(
            "Please install Flex Attention via `pip install torch` first"
        )

    o = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
    )

    o = torch.concat(
        (
            untile(o[:, :, :vision_seq_len, :], num_heads),  
            merge_heads(o[:, :, vision_seq_len:, :]),        
        ),
        dim=1, 
    )

    return o