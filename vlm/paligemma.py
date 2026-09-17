"""PaliGemma: SigLIP image encoder -> linear projector -> Gemma decoder.

Reference: Beyer et al., "PaliGemma: A versatile 3B VLM for transfer" (2024).
Image patch embeddings are projected to the LM width and *merged* into the text
embedding sequence at the ``<image>`` placeholder positions, so the decoder sees
a single multimodal token stream (prefix: image + prompt, bidirectional;
suffix: generated tokens, causal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from .gemma import GemmaConfig, GemmaForCausalLM, KVCache
from .siglip import SiglipVisionConfig, SiglipVisionModel


@dataclass
class PaliGemmaConfig:
    vision_config: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    text_config: GemmaConfig = field(default_factory=GemmaConfig)
    ignore_index: int = -100
    image_token_index: int = 257152
    vocab_size: int = 257216
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig(**self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = GemmaConfig(**self.text_config)
        self.text_config.pad_token_id = self.pad_token_id
        self.vocab_size = self.text_config.vocab_size
        self.vision_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim  # type: ignore[attr-defined]

    @classmethod
    def from_dict(cls, d: dict) -> "PaliGemmaConfig":
        """Build from a Hugging Face ``config.json`` dict, ignoring unknown keys."""
        import dataclasses

        vis = {k: v for k, v in d.get("vision_config", {}).items() if k in {f.name for f in dataclasses.fields(SiglipVisionConfig)}}
        txt = {k: v for k, v in d.get("text_config", {}).items() if k in {f.name for f in dataclasses.fields(GemmaConfig)}}
        top = {k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)} and k not in {"vision_config", "text_config"}}
        return cls(vision_config=SiglipVisionConfig(**vis), text_config=GemmaConfig(**txt), **top)


class PaliGemmaMultiModalProjector(nn.Module):
    """Linear map from SigLIP width to Gemma width (1152 -> 2048 for the 3B model)."""

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter projected image features into the ``<image>`` slots of the text
        embeddings and build the attention mask + position ids for this step."""
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Gemma applies sqrt(hidden) to *all* input embeddings; pre-divide image features
        # so they end up at their native scale after that multiplication.
        scaled_image_features = image_features / (self.config.text_config.hidden_size**0.5)

        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # masked_scatter fills image slots in order with the flattened image features
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features.to(dtype).reshape(-1)
        )
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # --- attention mask -------------------------------------------------
        # PaliGemma's prefix (image + prompt) is fully bidirectional, and during
        # generation each new token attends to everything before it, so with no
        # padding the additive mask is all zeros in both phases.
        q_len = inputs_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            if q_len != 1:
                raise ValueError("With a populated KV cache the query length must be 1")
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)
        causal_mask = causal_mask.unsqueeze(1)  # [B, 1, Sq, Skv] broadcast over heads

        # --- position ids ---------------------------------------------------
        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def encode_images(self, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # [B, C, H, W] -> [B, Np, D_vis] -> [B, Np, D_txt]
        image_features = self.vision_tower(pixel_values.to(dtype))
        return self.multi_modal_projector(image_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> dict:
        if not torch.all(attention_mask == 1):
            raise NotImplementedError("Padding is not supported; run one sample at a time")

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  # [B, S, D]

        if pixel_values is not None and (input_ids == self.config.image_token_index).any():
            image_features = self.encode_images(pixel_values, inputs_embeds.dtype)
            inputs_embeds, attention_mask_4d, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        else:
            # text-only step (e.g. decoding with a populated cache)
            inputs_embeds, attention_mask_4d, position_ids = self._merge_input_ids_with_image_features(
                torch.zeros(input_ids.shape[0], 0, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=inputs_embeds.device),
                inputs_embeds,
                input_ids,
                attention_mask,
                kv_cache,
            )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
