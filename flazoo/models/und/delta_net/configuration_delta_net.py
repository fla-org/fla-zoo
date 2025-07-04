# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class DeltaNetVisionConfig(PretrainedConfig):
    model_type = "delta_net_vision"

    def __init__(
        self,
        # DeltaNet core parameters
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_k: int = 1,
        expand_v: int = 1,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_beta: bool = True,
        use_output_norm: bool = True,
        num_heads: int = 16,
        qk_norm: str = "l2",
        qk_activation: str = "silu",
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 12,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        max_position_embeddings: int = 2048,
        attn_type: str = "full_attn",  # attention type, default to "full_attn"
        gradient_checkpointing: bool = False,
        compress_attention: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
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
        train_scan_type: str = "uni-scan",  # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        test_scan_type: str = None,  # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs,
    ):
        # Initialize DeltaNet core parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_beta = use_beta
        self.use_output_norm = use_output_norm
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.qk_activation = qk_activation
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.max_position_embeddings = max_position_embeddings
        self.attn_type = attn_type
        self.gradient_checkpointing = gradient_checkpointing
        self.compress_attention = compress_attention
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope

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
            if "layers" not in attn:
                raise ValueError(
                    "Layer indices must be provided to initialize hybrid attention layers"
                )
            if "num_heads" not in attn:
                raise ValueError(
                    "Number of heads must be provided to initialize hybrid attention layers"
                )
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", 10000.0)

        self.attn = attn

        if channel_mixer_dim is None:
            self.channel_mixer_dim = (
                4 * hidden_size
            )  # default value set to 4 * hidden_size
        else:
            self.channel_mixer_dim = channel_mixer_dim

        super().__init__(**kwargs)


class DeltaNetVideoConfig(PretrainedConfig):
    model_type = "delta_net_video"

    def __init__(
        self,
        # DeltaNet core parameters
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_k: int = 1,
        expand_v: int = 1,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_beta: bool = True,
        use_output_norm: bool = True,
        num_heads: int = 16,
        qk_norm: str = "l2",
        qk_activation: str = "silu",
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 12,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        max_position_embeddings: int = 2048,
        attn_type: str = "full_attn",  # attention type, default to "full_attn"
        gradient_checkpointing: bool = False,
        compress_attention: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        # Video specific parameters
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
        train_scan_type: str = "uni-scan",  # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        test_scan_type: str = None,  # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        norm_pix_loss: bool = True,
        num_frames: int = 16,
        tubelet_size: int = 2,
        t_dim: int = 32,
        h_dim: int = 16,
        w_dim: int = 16,
        # decoder specific parameters
        decoder_num_heads: int = 6,
        decoder_hidden_size: int = 256,
        decoder_num_hidden_layers: int = 4,
        decoder_channel_mixer_dim: int = None,
        **kwargs,
    ):
        # Initialize DeltaNet core parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_beta = use_beta
        self.use_output_norm = use_output_norm
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.qk_activation = qk_activation
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.max_position_embeddings = max_position_embeddings
        self.attn_type = attn_type
        self.gradient_checkpointing = gradient_checkpointing
        self.compress_attention = compress_attention
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope

        # Initialize video specific parameters
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
        self.norm_pix_loss = norm_pix_loss
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        # Initialize decoder specific parameters
        self.decoder_num_heads = decoder_num_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if "layers" not in attn:
                raise ValueError(
                    "Layer indices must be provided to initialize hybrid attention layers"
                )
            if "num_heads" not in attn:
                raise ValueError(
                    "Number of heads must be provided to initialize hybrid attention layers"
                )
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", 10000.0)

        self.attn = attn

        if channel_mixer_dim is None:
            self.channel_mixer_dim = (
                4 * hidden_size
            )  # default value set to 4 * hidden_size
        else:
            self.channel_mixer_dim = channel_mixer_dim

        if decoder_channel_mixer_dim is None:
            self.decoder_channel_mixer_dim = 4 * decoder_hidden_size
        else:
            self.decoder_channel_mixer_dim = decoder_channel_mixer_dim  # default value set to 4 * decoder_hidden_size

        super().__init__(**kwargs)
