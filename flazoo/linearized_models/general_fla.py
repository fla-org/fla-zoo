import logging
from torch import nn

from fla.layers import (
    DeltaNet, 
    GatedDeltaNet,
    ForgettingAttention,
    ABCAttention,
    GatedLinearAttention,
    BitAttention,
    GatedSlotAttention,
    HGRNAttention,
    HGRN2Attention,
    LightNetAttention,
    LinearAttention,
    MultiScaleRetention,
    RWKV6Attention,
    RWKV7Attention
)

fla_attn_mapping = {
    "deltanet" : DeltaNet,
    "gated_deltanet" : GatedDeltaNet,
    "fox" : ForgettingAttention,
    "abc" : ABCAttention,
    "gla" : GatedLinearAttention,
    "bitnet" : BitAttention,
    "gsa" : GatedSlotAttention,
    "hgrn" : HGRNAttention,
    "hgrn2" : HGRN2Attention,
    "lightnet" : LightNetAttention,
    "linear_attention" : LinearAttention,
    "retnet" : MultiScaleRetention,
    "rwkv6" : RWKV6Attention,
    "rwkv7" : RWKV7Attention
}

class GeneralizedFlashLinearAttention(nn.Module):
    def __init__(self, fla_config, layer_idx: int = None):
        super().__init__()
        logging.info(f"Using {fla_config["fla_type"]} attention")
        
        self.attn = fla_attn_mapping[fla_config["fla_type"]](**fla_config)
    
    def forward(self, hidden_states, **kwargs):
        return self.attn(hidden_states, **kwargs)
