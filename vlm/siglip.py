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


class SiglipAttention(nn.Module):
    """Standard multi-head self-attention (no masking: every patch sees every patch)."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.size()
        # [B, N, D] -> [B, H, N, Dh]
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, H, N, Dh] @ [B, H, Dh, N] -> [B, H, N, N]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # [B, H, N, N] @ [B, H, N, Dh] -> [B, H, N, Dh]
        attn_output = torch.matmul(attn_weights, v)
        # [B, H, N, Dh] -> [B, N, H, Dh] -> [B, N, D]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_output), attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        return self.fc2(hidden_states)


class SiglipEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN1(x)); x + MLP(LN2(x))."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, Np, D]
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)


class SiglipVisionModel(nn.Module):
    """Top-level wrapper; matches the ``vision_tower.vision_model`` checkpoint prefix."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns one contextualised embedding per patch: [B, num_patches, hidden_size]."""
        return self.vision_model(pixel_values)
