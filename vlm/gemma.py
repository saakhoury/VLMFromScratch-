"""Gemma decoder-only language model, written from scratch.

Reference: Gemma Team, "Gemma: Open Models Based on Gemini Research and Technology" (2024).
Implements RMSNorm (Gemma's ``1 + weight`` variant), rotary position embeddings,
grouped-query attention with a concatenation-based KV cache, GeGLU MLP and
weight tying between the token embedding and the LM head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class GemmaConfig:
    vocab_size: int = 257216
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")


class GemmaRMSNorm(nn.Module):
    """Root-mean-square LayerNorm. Gemma parameterises the scale as ``1 + weight``
    (weight is zero-initialised) and normalises in float32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    """Rotary position embeddings (Su et al., RoFormer, 2021).

    Pairs dimension ``i`` with ``i + head_dim/2`` (the "rotate_half" convention used
    by the released checkpoints) and rotates each pair by ``pos * theta^(-2i/d)``.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # theta_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is only used for dtype/device. position_ids: [B, S]
        inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # [B, d/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, S]
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # [B, d/2, 1] @ [B, 1, S] -> [B, d/2, S] -> [B, S, d/2]
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, d]
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [B, S, d] -> [B, 1, S, d] to broadcast over heads
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class KVCache:
    """Per-layer key/value cache grown by ``torch.cat`` along the sequence axis.

    Growing by concatenation means the cache is always exactly as long as the
    number of tokens seen so far. A pre-allocated cache of ``max_position_embeddings``
    would reserve ``B * L * H_kv * S_max * Dh`` per K and V regardless of prompt
    length (see ``scripts/benchmark_kv_cache.py`` for the measured saving).
    """

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        # [B, H_kv, S, Dh]
        return self.key_cache[0].shape[-2]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # first time this layer is seen (prefill): store as-is
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.key_cache + self.value_cache)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, S, Dh] -> [B, H_kv * n_rep, S, Dh] so each query group shares one KV head."""
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GemmaAttention(nn.Module):
    """Grouped-query attention (Ainslie et al., 2023): ``num_attention_heads`` query
    heads share ``num_key_value_heads`` key/value heads (Gemma-2B uses 8 -> 1, i.e. MQA).
    Fewer KV heads shrink both the projection weights and the KV cache."""

    def __init__(self, config: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        # [B, S, D] -> [B, H, S, Dh] / [B, H_kv, S, Dh]
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # broadcast the KV heads across their query group
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # [B, H, Sq, Dh] @ [B, H, Dh, Skv] -> [B, H, Sq, Skv]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)  # [B, H, Sq, Dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)  # [B, Sq, H*Dh]
        return self.o_proj(attn_output), attn_weights
