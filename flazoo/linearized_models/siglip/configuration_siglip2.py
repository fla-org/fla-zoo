from typing import Optional, Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class FLASiglip2TextConfig(PretrainedConfig):

    model_type = "fla_siglip2_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # This differs from `CLIPTokenizer`'s default and from openai/siglip2
        # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        projection_size=None,
        fla: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.projection_size = projection_size if projection_size is not None else hidden_size
        self.fla = fla


class FLASiglip2VisionConfig(PretrainedConfig):

    model_type = "fla_siglip2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        fla: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_patches = num_patches
        self.fla = fla


class FLASiglip2Config(PretrainedConfig):

    model_type = "fla_siglip2"
    sub_configs = {"text_config": FLASiglip2TextConfig, "vision_config": FLASiglip2VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `Siglip2TextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `Siglip2VisionConfig` with default values.")

        self.text_config = FLASiglip2TextConfig(**text_config)
        self.vision_config = FLASiglip2VisionConfig(**vision_config)

        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: FLASiglip2TextConfig, vision_config: FLASiglip2VisionConfig, **kwargs):
        r"""
        Instantiate a [`Siglip2Config`] (or a derived class) from siglip2 text model configuration and siglip2 vision
        model configuration.

        Returns:
            [`Siglip2Config`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


__all__ = ["FLASiglip2Config", "FLASiglip2TextConfig", "FLASiglip2VisionConfig"]
