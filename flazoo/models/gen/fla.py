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

class FLAAttentionWrapper(nn.Module):
    """
    A wrapper class for FLA attention modules. You can use various FLA attention modules by specifying the `fla_type` parameter.
    under the hood, this class will instantiate the specified FLA attention module and pass the input to it.
    """

    pass