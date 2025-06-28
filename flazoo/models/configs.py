# -*- coding: utf-8 -*-

from typing import Dict, Optional, Any

from transformers.configuration_utils import PretrainedConfig

FLA_ARGS_DEFAULTS = {
    "deltanet": {
        "use_gate": False,
        "use_output_norm": True,
        "qk_norm": "l2",
        "qk_activation": "silu",
    },
    "gated_deltanet": {
        "use_gate": True,
    },
    "gated_deltaproduct": {
        "use_gate": True,
        "use_forget_gate": False,
        "allow_neg_eigval": False,
        "num_householder": 1,
    },
    "bitnet": {
        "rope_theta": 10000.0,
        "elementwise_affine": True,
        "attention_bias": False,
        "fuse_norm": True,
        "qkv_bias": True,
    },
    "retnet": {
        "use_short_conv": False,
        "use_output_gate": True,
        "elementwise_affine": True,
        "fuse_norm": True,
        "qkv_bias": True,
    },
    "mesa_net": {
        "lambda_lower_bound": 0.25,
        "fuse_norm": True,
        "fuse_swiglu": True,
        "max_cg_step_training": 30,
        "max_cg_step_decoding": 30,
    },
    "linear_attn": {
        "feature_map": "elementwise_product",
        "elementwise_affine": True,
        "qkv_bias": True,
    },
    "lightnet": {
        "expand_ratio": 128,
        "use_short_conv": False,
        "gate_low_rank_dim": 128,
        "elementwise_affine": True,
        "fuse_norm": True,
        "fuse_swiglu": True,
    },
    "lact": {
        "use_short_conv": True,
    },
    "hgrn": {
        "expand_ratio": 1,
        "use_short_conv": True,
        "use_lower_bound": True,
        "elementwise_affine": True,
        "qkv_bias": True,
    },
    "hgrn2": {
        "expand_ratio": 128,
        "use_short_conv": True,
        "use_lower_bound": True,
        "elementwise_affine": True,
        "qkv_bias": True,
    },
    "gsa": {
        "gate_logit_normalizer": 8,
        "num_slots": 64,
        "use_short_conv": False,
        "feature_map": "swish",
        "use_norm": True,
        "norm_first": True,
        "elementwise_affine": True,
        "fuse_norm": True,
        "qkv_bias": True,
    },
    "abc": {
        "gate_low_rank_dim": 16,
        "clamp_min": -32.0,
        "clamp_max": 32.0,
        "num_slots": 64,
        "use_short_conv": False,
        "elementwise_affine": True,
        "fuse_norm": True,
        "qkv_bias": True,
    },
    "gla": {
        "use_short_conv": False,
        "use_output_gate": True,
        "elementwise_affine": True,
        "use_gk": True,
        "fuse_norm": True,
    },
}


class FLAVisionConfig(PretrainedConfig):
    model_type = "fla_vision"

    def __init__(
        self,
        # Default set to DeltaNet
        fla_attn_type: str = "deltanet",
        hidden_size: int = 2048,
        num_hidden_layers: int = 12,
        num_heads: int = 16,
        hidden_act: str = "swish",
        attn_mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        fuse_cross_entropy: bool = True,
        gradient_checkpointing: bool = False,
        attn_type: str = "full_attn",
        attn: Optional[Dict] = None,
        expand_k: float = 1,
        expand_v: float = 1,
        use_gate: bool = None,
        use_beta: bool = True,
        use_output_norm: bool = True,
        qk_norm: str = None,
        qk_activation: str = None,
        intermediate_size: Optional[int] = None,
        norm_first: bool = False,
        compress_attention: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        num_kv_heads: int = None,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = None,
        elementwise_affine: Optional[bool] = None,
        attention_bias: bool = None,
        fuse_norm: bool = None,
        qkv_bias: bool = None,
        feature_map: Optional[str] = None,
        use_output_gate: bool = None,
        lambda_lower_bound: float = None,
        fuse_swiglu: bool = None,
        max_cg_step_training: int = None,
        max_cg_step_decoding: int = None,
        tie_feature_map_qk: bool = None,
        norm_q: bool = None,
        norm_k: bool = None,
        norm_feature_map: bool = None,
        expand_ratio: Optional[int] = None,
        gate_low_rank_dim: int = None,
        use_lower_bound: bool = None,
        clamp_min: float = None,
        clamp_max: float = None,
        num_slots: Optional[int] = None,
        gate_logit_normalizer: Optional[int] = None,
        use_norm: bool = None,
        use_forget_gate: bool = None,
        allow_neg_eigval: bool = None,
        num_householder: int = None,
        use_gk: bool = None,
        use_gv: bool = None,
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
        train_scan_type: str = "uni-scan",
        test_scan_type: str = None,
        **kwargs,
    ):
        # Get the default values for the chosen model variant
        defaults = FLA_ARGS_DEFAULTS.get(fla_attn_type)
        if defaults is None:
            raise ValueError(
                f"Unknown fla_attn_type: '{fla_attn_type}'. "
                f"Available options: {list(FLA_ARGS_DEFAULTS.keys())}"
            )

        self.fla_attn_type = fla_attn_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.fuse_cross_entropy = fuse_cross_entropy
        self.gradient_checkpointing = gradient_checkpointing
        self.attn_type = attn_type
        self.attn = attn
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_gate = use_gate if use_gate is not None else defaults.get("use_gate")
        self.use_beta = use_beta
        self.use_output_norm = use_output_norm
        self.qk_norm = qk_norm if qk_norm is not None else defaults.get("qk_norm")
        self.qk_activation = (
            qk_activation
            if qk_activation is not None
            else defaults.get("qk_activation")
        )
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first
        self.compress_attention = compress_attention
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.rope_theta = (
            rope_theta if rope_theta is not None else defaults.get("rope_theta")
        )
        self.elementwise_affine = (
            elementwise_affine
            if elementwise_affine is not None
            else defaults.get("elementwise_affine")
        )
        self.attention_bias = (
            attention_bias
            if attention_bias is not None
            else defaults.get("attention_bias")
        )
        self.fuse_norm = (
            fuse_norm if fuse_norm is not None else defaults.get("fuse_norm")
        )
        self.qkv_bias = qkv_bias if qkv_bias is not None else defaults.get("qkv_bias")
        self.feature_map = (
            feature_map if feature_map is not None else defaults.get("feature_map")
        )
        self.use_output_gate = (
            use_output_gate
            if use_output_gate is not None
            else defaults.get("use_output_gate")
        )
        self.lambda_lower_bound = (
            lambda_lower_bound
            if lambda_lower_bound is not None
            else defaults.get("lambda_lower_bound")
        )
        self.fuse_swiglu = (
            fuse_swiglu if fuse_swiglu is not None else defaults.get("fuse_swiglu")
        )
        self.max_cg_step_training = (
            max_cg_step_training
            if max_cg_step_training is not None
            else defaults.get("max_cg_step_training")
        )
        self.max_cg_step_decoding = (
            max_cg_step_decoding
            if max_cg_step_decoding is not None
            else defaults.get("max_cg_step_decoding")
        )
        self.tie_feature_map_qk = tie_feature_map_qk
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_feature_map = norm_feature_map
        self.expand_ratio = (
            expand_ratio if expand_ratio is not None else defaults.get("expand_ratio")
        )
        self.gate_low_rank_dim = (
            gate_low_rank_dim
            if gate_low_rank_dim is not None
            else defaults.get("gate_low_rank_dim")
        )
        self.use_lower_bound = (
            use_lower_bound
            if use_lower_bound is not None
            else defaults.get("use_lower_bound")
        )
        self.clamp_min = (
            clamp_min if clamp_min is not None else defaults.get("clamp_min")
        )
        self.clamp_max = (
            clamp_max if clamp_max is not None else defaults.get("clamp_max")
        )
        self.num_slots = (
            num_slots if num_slots is not None else defaults.get("num_slots")
        )
        self.gate_logit_normalizer = (
            gate_logit_normalizer
            if gate_logit_normalizer is not None
            else defaults.get("gate_logit_normalizer")
        )
        self.use_norm = use_norm if use_norm is not None else defaults.get("use_norm")
        self.use_forget_gate = (
            use_forget_gate
            if use_forget_gate is not None
            else defaults.get("use_forget_gate")
        )
        self.allow_neg_eigval = (
            allow_neg_eigval
            if allow_neg_eigval is not None
            else defaults.get("allow_neg_eigval")
        )
        self.num_householder = (
            num_householder
            if num_householder is not None
            else defaults.get("num_householder")
        )
        self.use_gk = use_gk if use_gk is not None else defaults.get("use_gk")
        self.use_gv = use_gv
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_mask_token = use_mask_token
        self.layer_norm_eps = layer_norm_eps
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.encoder_stride = encoder_stride
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


class FLAVideoConfig(PretrainedConfig):
    model_type = "fla_video"

    def __init__(
        self,
        # Default set to DeltaNet
        fla_attn_type: str = "deltanet",
        hidden_size: int = 2048,
        num_hidden_layers: int = 12,
        num_heads: int = 16,
        hidden_act: str = "swish",
        attn_mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        fuse_cross_entropy: bool = True,
        gradient_checkpointing: bool = False,
        attn_type: str = "full_attn",
        attn: Optional[Dict] = None,
        expand_k: float = 1,
        expand_v: float = 1,
        use_gate: bool = None,
        use_beta: bool = True,
        use_output_norm: bool = None,
        qk_norm: str = None,
        qk_activation: str = None,
        intermediate_size: Optional[int] = None,
        norm_first: bool = False,
        compress_attention: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        num_kv_heads: int = None,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = None,
        elementwise_affine: Optional[bool] = None,
        attention_bias: bool = None,
        fuse_norm: bool = None,
        qkv_bias: bool = None,
        feature_map: Optional[str] = None,
        use_output_gate: bool = None,
        lambda_lower_bound: float = None,
        fuse_swiglu: bool = None,
        max_cg_step_training: int = None,
        max_cg_step_decoding: int = None,
        tie_feature_map_qk: bool = None,
        norm_q: bool = None,
        norm_k: bool = None,
        norm_feature_map: bool = None,
        expand_ratio: Optional[int] = None,
        gate_low_rank_dim: int = None,
        use_lower_bound: bool = None,
        clamp_min: float = None,
        clamp_max: float = None,
        num_slots: Optional[int] = None,
        gate_logit_normalizer: Optional[int] = None,
        use_norm: bool = None,
        use_forget_gate: bool = None,
        allow_neg_eigval: bool = None,
        num_householder: int = None,
        use_gk: bool = None,
        use_gv: bool = None,
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
        decoder_num_heads: int = 6,
        decoder_hidden_size: int = 256,
        decoder_num_hidden_layers: int = 4,
        decoder_channel_mixer_dim: int = None,
        **kwargs,
    ):
        # Get the default values for the chosen model variant
        defaults = FLA_ARGS_DEFAULTS.get(fla_attn_type)
        if defaults is None:
            raise ValueError(
                f"Unknown fla_attn_type: '{fla_attn_type}'. "
                f"Available options: {list(FLA_ARGS_DEFAULTS.keys())}"
            )

        self.fla_attn_type = fla_attn_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.fuse_cross_entropy = fuse_cross_entropy
        self.gradient_checkpointing = gradient_checkpointing
        self.attn_type = attn_type
        self.attn = attn
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_gate = use_gate if use_gate is not None else defaults.get("use_gate")
        self.use_beta = use_beta
        self.use_output_norm = use_output_norm
        self.qk_norm = qk_norm if qk_norm is not None else defaults.get("qk_norm")
        self.qk_activation = (
            qk_activation
            if qk_activation is not None
            else defaults.get("qk_activation")
        )
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first
        self.compress_attention = compress_attention
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.rope_theta = (
            rope_theta if rope_theta is not None else defaults.get("rope_theta")
        )
        self.elementwise_affine = (
            elementwise_affine
            if elementwise_affine is not None
            else defaults.get("elementwise_affine")
        )
        self.attention_bias = (
            attention_bias
            if attention_bias is not None
            else defaults.get("attention_bias")
        )
        self.fuse_norm = (
            fuse_norm if fuse_norm is not None else defaults.get("fuse_norm")
        )
        self.qkv_bias = qkv_bias if qkv_bias is not None else defaults.get("qkv_bias")
        self.feature_map = (
            feature_map if feature_map is not None else defaults.get("feature_map")
        )
        self.use_output_gate = (
            use_output_gate
            if use_output_gate is not None
            else defaults.get("use_output_gate")
        )
        self.lambda_lower_bound = (
            lambda_lower_bound
            if lambda_lower_bound is not None
            else defaults.get("lambda_lower_bound")
        )
        self.fuse_swiglu = (
            fuse_swiglu if fuse_swiglu is not None else defaults.get("fuse_swiglu")
        )
        self.max_cg_step_training = (
            max_cg_step_training
            if max_cg_step_training is not None
            else defaults.get("max_cg_step_training")
        )
        self.max_cg_step_decoding = (
            max_cg_step_decoding
            if max_cg_step_decoding is not None
            else defaults.get("max_cg_step_decoding")
        )
        self.tie_feature_map_qk = tie_feature_map_qk
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_feature_map = norm_feature_map
        self.expand_ratio = (
            expand_ratio if expand_ratio is not None else defaults.get("expand_ratio")
        )
        self.gate_low_rank_dim = (
            gate_low_rank_dim
            if gate_low_rank_dim is not None
            else defaults.get("gate_low_rank_dim")
        )
        self.use_lower_bound = (
            use_lower_bound
            if use_lower_bound is not None
            else defaults.get("use_lower_bound")
        )
        self.clamp_min = (
            clamp_min if clamp_min is not None else defaults.get("clamp_min")
        )
        self.clamp_max = (
            clamp_max if clamp_max is not None else defaults.get("clamp_max")
        )
        self.num_slots = (
            num_slots if num_slots is not None else defaults.get("num_slots")
        )
        self.gate_logit_normalizer = (
            gate_logit_normalizer
            if gate_logit_normalizer is not None
            else defaults.get("gate_logit_normalizer")
        )
        self.use_norm = use_norm if use_norm is not None else defaults.get("use_norm")
        self.use_forget_gate = (
            use_forget_gate
            if use_forget_gate is not None
            else defaults.get("use_forget_gate")
        )
        self.allow_neg_eigval = (
            allow_neg_eigval
            if allow_neg_eigval is not None
            else defaults.get("allow_neg_eigval")
        )
        self.num_householder = (
            num_householder
            if num_householder is not None
            else defaults.get("num_householder")
        )
        self.use_gk = use_gk if use_gk is not None else defaults.get("use_gk")
        self.use_gv = use_gv
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
