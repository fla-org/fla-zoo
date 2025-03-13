# -*- coding: utf-8 -*-

from typing import Dict, Optional, Union, List

from transformers.configuration_utils import PretrainedConfig

class RWKV7VisionConfig(PretrainedConfig):

    model_type = 'rwkv7_vision'

    def __init__(
        self,
        # RWKV6 core parameters
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        hidden_act: str = "sqrelu",
        norm_first: bool = True,
        norm_bias: bool = True,
        norm_eps: float = 1e-5,
        attn: Optional[Dict] = None,
        initializer_range: float = 0.006,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        value_dim: Optional[Union[int, List[int]]] = None,
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
        train_scan_type: str = "uni-scan", # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        test_scan_type: str = None, # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs
    ):
        # Initialize RWKV6 core parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.hidden_act = hidden_act
        self.norm_first = norm_first
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.value_dim = value_dim
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.value_dim = value_dim
        

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
            attn['window_size'] = attn.get('window_size', None)

        self.attn = attn

        if mlp_dim is None:
            self.mlp_dim = 4 * hidden_size # default value set to 4 * hidden_size
        else:
            self.mlp_dim = mlp_dim
        
        super().__init__(**kwargs)