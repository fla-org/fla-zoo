# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig

class ABCVisionConfig(PretrainedConfig):

    model_type = 'abc_vision'

    def __init__(
        self,
        # ABC core parameters
        hidden_size: int = 2048,
        gate_low_rank_dim: int = 16,
        clamp_min: float = -32,
        clamp_max: float = 32,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        num_slots: Optional[int] = 64,
        use_short_conv: bool = False,
        conv_size: int = 4,
        exapnd_k: float = 0.5,
        exapnd_v: float = 1,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        # Vision specific parameters
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 1000,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.0,
        use_mask_token: bool = False,
        layer_norm_eps: float = 1e-6,
        interpolate_pos_encoding: bool = False,
        mlp_dim: int = None,
        encoder_stride=16,
        scan_type: str = "uni-scan", # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs
    ):
        # Initialize ABC core parameters
        self.hidden_size = hidden_size
        self.gate_low_rank_dim = gate_low_rank_dim
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.expand_k = exapnd_k
        self.expand_v = exapnd_v
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy

        # Initialize vision specific parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_mask_token = use_mask_token
        self.layer_norm_eps = layer_norm_eps
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.scan_type = scan_type
        self.encoder_stride = encoder_stride

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['window_size'] = attn.get('window_size', None)

        self.attn = attn
        if mlp_dim is None:
            self.mlp_dim = 4 * hidden_size # default value set to 4 * hidden_size
        else:
            self.mlp_dim = mlp_dim
        
        super().__init__(**kwargs)