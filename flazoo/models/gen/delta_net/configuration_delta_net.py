from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class DeltaNetGen2DConfig(PretrainedConfig):
    model_type = "delta_net_gen_2d"

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
        # Gen2D specific parameters
        path_type="edm",
        input_size=32,
        patch_size=2,
        in_channels=4,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=12,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        channel_mixer_dim=2048,
        projection_dim=2048,
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

        # Initialize Gen2D specific parameters
        self.path_type = path_type
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_depth = encoder_depth
        self.depth = depth
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.use_cfg = use_cfg
        self.z_dims = z_dims
        self.channel_mixer_dim = channel_mixer_dim
        self.projection_dim = projection_dim

        self.train_scan_type = train_scan_type

        if test_scan_type is None:
            self.test_scan_type = train_scan_type
        else:
            self.test_scan_type = test_scan_type

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

        super().__init__(**kwargs)
