# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig

class XAttentionVisionConfig(PretrainedConfig):

    model_type = 'xattn_vision'

    def __init__(
        self,
        # XAttn core parameters
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 64,
        num_kv_heads: int = 4,
        window_size: int = 32,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: int = 2048,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: Optional[bool] = True,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        attention_bias: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        stride: int = 16,
        block_size: int = 128,
        chunk_size: int = 2048,

        # Vision specific parameters
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        num_classes: int = 1000,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.0,
        use_mask_token: bool = False,
        layer_norm_eps: float = 1e-6,
        interpolate_pos_encoding: bool = False,
        channel_mixer_dim: int = None,
        encoder_stride=16,
        train_scan_type: str = "uni-scan", # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        test_scan_type: str = None, # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs
    ):
        # Initialize Xattn core parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.stride = stride

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

        if channel_mixer_dim is None:
            self.channel_mixer_dim = 4 * hidden_size # default value set to 4 * hidden_size
        else:
            self.channel_mixer_dim = channel_mixer_dim
        
        super().__init__(**kwargs)