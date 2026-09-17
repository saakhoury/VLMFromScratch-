"""Config presets.

``paligemma_3b_224`` mirrors ``google/paligemma-3b-pt-224`` so released SafeTensors
load directly. ``paligemma_small`` is a 12-layer decoder with a 6-layer SigLIP
encoder that fits on a laptop GPU/MPS for from-scratch experiments and tests.
"""

from __future__ import annotations

from .gemma import GemmaConfig
from .paligemma import PaliGemmaConfig
from .siglip import SiglipVisionConfig

# Special token ids shared by all PaliGemma checkpoints (Gemma tokenizer + 1152 extra tokens)
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 2
IMAGE_TOKEN_ID = 257152
VOCAB_SIZE = 257216


def paligemma_3b_224() -> PaliGemmaConfig:
    return PaliGemmaConfig(
        vision_config=SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
        ),
        text_config=GemmaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=2048,
            intermediate_size=16384,
            num_hidden_layers=18,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
        ),
        image_token_index=IMAGE_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=PAD_TOKEN_ID,
    )


def paligemma_small(vocab_size: int = VOCAB_SIZE) -> PaliGemmaConfig:
    """~12-layer, 512-wide decoder with 4 query heads sharing 1 KV head."""
    return PaliGemmaConfig(
        vision_config=SiglipVisionConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=4,
            image_size=224,
            patch_size=14,
        ),
        text_config=GemmaConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
        ),
        image_token_index=IMAGE_TOKEN_ID if vocab_size == VOCAB_SIZE else vocab_size - 64,
        vocab_size=vocab_size,
        projection_dim=512,
        hidden_size=512,
        pad_token_id=PAD_TOKEN_ID,
    )


def paligemma_tiny() -> PaliGemmaConfig:
    """Unit-test sized model (runs in milliseconds on CPU)."""
    return PaliGemmaConfig(
        vision_config=SiglipVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=16, patch_size=8
        ),
        text_config=GemmaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        ),
        image_token_index=200,
        vocab_size=256,
        projection_dim=64,
        hidden_size=64,
        pad_token_id=0,
    )
