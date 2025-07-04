# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class GatedDeltaProductVisionConfig(PretrainedConfig):
    model_type = "gated_deltaproduct_vision"

    def __init__(
        self,
        # GatedDeltaProduct core parameters
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: int = 1,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        max_position_embeddings: int = 2048,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        use_forget_gate: bool = False,
        allow_neg_eigval: bool = False,
        num_householder: int = 1,
        attn_type: str = "full_attn",  # attention type, default to "full_attn"
        gradient_checkpointing: bool = False,
        # Vision specific parameters
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 1000,
        hidden_dropout_prob: float = 0.0,
        use_mask_token: bool = False,
        layer_norm_eps: float = 1e-6,
        interpolate_pos_encoding: bool = False,
        channel_mixer_dim: int = None,
        encoder_stride=16,
        train_scan_type: str = "uni-scan",  # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        test_scan_type: str = None,  # scaning type, "uni-scan" or "bi-scan" or "cross-scan", default to "uni-scan"
        **kwargs,
    ):
        # Initialize GatedDeltaProduct core parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_forget_gate = use_forget_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.num_householder = num_householder
        self.attn = attn
        self.max_position_embeddings = max_position_embeddings
        self.attn_type = attn_type
        self.gradient_checkpointing = gradient_checkpointing

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

        self.attn = attn

        if channel_mixer_dim is None:
            self.channel_mixer_dim = 4 * hidden_size
        else:
            self.channel_mixer_dim = channel_mixer_dim

        super().__init__(**kwargs)


class GatedDeltaProductVideoConfig(PretrainedConfig):
    model_type = "gated_deltaproduct_video"

    def __init__(
        self,
        # GatedDeltaProduct core parameters
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: int = 1,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        max_position_embeddings: int = 2048,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        use_forget_gate: bool = False,
        allow_neg_eigval: bool = False,
        num_householder: int = 1,
        attn_type: str = "full_attn",  # attention type, default to "full_attn"
        gradient_checkpointing: bool = False,
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
        # decoder specific parameters
        decoder_num_heads: int = 6,
        decoder_hidden_size: int = 256,
        decoder_num_hidden_layers: int = 4,
        decoder_channel_mixer_dim: int = None,
        **kwargs,
    ):
        # Initialize GatedDeltaProduct core parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_forget_gate = use_forget_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.num_householder = num_householder
        self.attn = attn
        self.max_position_embeddings = max_position_embeddings
        self.attn_type = attn_type
        self.gradient_checkpointing = gradient_checkpointing

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
