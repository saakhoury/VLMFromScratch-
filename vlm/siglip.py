"""SigLIP vision encoder, written from scratch.

Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (2023).
Parameter names mirror the Hugging Face checkpoint layout so PaliGemma
SafeTensors shards load directly (``vision_tower.vision_model.*``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_image_tokens is None:
            self.num_image_tokens = (self.image_size // self.patch_size) ** 2

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class SiglipVisionEmbeddings(nn.Module):
    """Non-overlapping conv patchify + learned absolute position embeddings."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",  # no padding: image_size must be a multiple of patch_size
        )
        self.num_patches = config.num_patches
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W] -> [B, D, H/P, W/P]
        patch_embeds = self.patch_embedding(pixel_values)
        # [B, D, Hp, Wp] -> [B, D, Np] -> [B, Np, D]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding(self.position_ids)
