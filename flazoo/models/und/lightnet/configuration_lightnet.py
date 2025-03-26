
# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class LightNetVisionConfig(PretrainedConfig):

    model_type = 'lightnet_vision'

    def __init__(
        self,
         # LightNet core parameters
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        attn_mode: str = "chunk",
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        hidden_act: str = "swish",
        gate_low_rank_dim: int = 128,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        attn_type: str = "full_attn", # attention type, default to "full_attn"

        # Vision specific parameters
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 1000,
        hidden_dropout_prob: float = 0.0,
        use_mask_token: bool = False,
        layer_norm_eps: float = 1e-6,
        interpolate_pos_encoding: bool = False,
        encoder_stride=16,
        channel_mixer_dim: int = None,
        train_scan_type: str = "uni-scan", # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        test_scan_type: str = None, # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs
    ):
        # Initialize LightNet core parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.gate_low_rank_dim = gate_low_rank_dim
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy

        self.attn_type = attn_type

        # Initialize vision specific parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_mask_token = use_mask_token
        self.layer_norm_eps = layer_norm_eps
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.train_scan_type = train_scan_type
        
        if test_scan_type is None:
            self.test_scan_type = train_scan_type
        else:
            self.test_scan_type = test_scan_type
        self.encoder_stride = encoder_stride

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        self.attn = attn

        if channel_mixer_dim is None:
            self.channel_mixer_dim = 4 * hidden_size # default value set to 4 * hidden_size
        else:
            self.channel_mixer_dim = channel_mixer_dim
        
        super().__init__(**kwargs)
