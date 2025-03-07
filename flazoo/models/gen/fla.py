from fla.layers.abc import ABCAttention
from fla.layers.bitattn import BitAttention
from fla.layers.delta_net import DeltaNet
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gla import GatedLinearAttention
from fla.layers.gsa import GatedSlotAttention
from fla.layers.hgrn import HGRNAttention
from fla.layers.hgrn2 import HGRN2Attention
from fla.layers.lightnet import LightNetAttention
from fla.layers.linear_attn import LinearAttention
from fla.layers.multiscale_retention import MultiScaleRetention
from fla.layers.rwkv6 import RWKV6Attention
from fla.layers.rwkv7 import RWKV7Attention
from fla.layers.simple_gla import SimpleGatedLinearAttention
import torch.nn as nn
from flazoo.models.utils import prepare_hidden_states_for_scan, prepare_hidden_states_for_merge

# A dictionary that maps the fla_type to the corresponding FLA attention module
FLA_ATTN_MAPS = {
    'bitnet': BitAttention,
    'deltanet': DeltaNet,
    'gated_deltanet': GatedDeltaNet,
    'gla': GatedLinearAttention,
    'gsa': GatedSlotAttention,
    'hgrn': HGRNAttention,
    'hgrn2': HGRN2Attention,
    'lightnet': LightNetAttention,
    'linear_attn': LinearAttention,
    'retnet': MultiScaleRetention,
    'rwkv6': RWKV6Attention,
    'rwkv7': RWKV7Attention,
}

class FLAAttentionWrapper(nn.Module):
    """
    A wrapper class for FLA attention modules. You can use various FLA attention modules by specifying the `fla_type` parameter.
    under the hood, this class will instantiate the specified FLA attention module.
    Before pass the input to the attention module, this class will prepare the hidden states by calling `prepare_hidden_states_for_scan`.
    After the attention module is applied, this class will prepare the hidden states by calling `prepare_hidden_states_for_merge`.
    """
    def __init__(self, fla_type: str, scan_type: str, fla_config: dict, **kwargs):
        super().__init__()

        assert fla_type in FLA_ATTN_MAPS, f"FLA type {fla_type} is not supported. Supported types are {list(FLA_ATTN_MAPS.keys())}"

        self.fla_config = fla_config

        self.attn = FLA_ATTN_MAPS[fla_type](**fla_config)

    def forward(self, hidden_states, output_attentions):
        hidden_states = prepare_hidden_states_for_scan(hidden_states, self.fla_config)
        hidden_states = self.attn(hidden_states, output_attentions=output_attentions)
        hidden_states = prepare_hidden_states_for_merge(hidden_states, self.fla_config)

        return hidden_states


