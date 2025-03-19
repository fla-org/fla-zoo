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
    
def prepare_hidden_states_for_scan(hidden_states: torch.Tensor, train_scan_type: str = "uni-scan", test_scan_type : str = "uni-scan", training: bool = True):
    scan_type = train_scan_type if training else test_scan_type
    # hidden_states shape should be: (B, L, D)
    if scan_type == "uni-scan":
        return hidden_states
    elif scan_type == "random-scan":
        L = hidden_states.size(1)
        random_idx = torch.randperm(L, device=hidden_states.device)
        hidden_states = hidden_states[:, random_idx, :]
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

def prepare_hidden_states_for_merge(hidden_states: torch.Tensor, train_scan_type: str = "uni-scan", test_scan_type : str = "uni-scan", training: bool = True, layer_idx: Optional[int] = None):
    scan_type = train_scan_type if training else test_scan_type    
    # hidden_states shape should be: (BK, L, D), K=2 for bi-scan, K=1 for uni-scan, K=4 for cross-scan
    if scan_type == "uni-scan" or scan_type == "random-scan" or scan_type == "flip-scan":
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