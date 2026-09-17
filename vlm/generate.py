"""Autoregressive decoding with KV cache and nucleus (top-p) sampling."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .gemma import KVCache
from .paligemma import PaliGemmaForConditionalGeneration


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling (Holtzman et al., 2020): keep the smallest set of tokens
    whose cumulative probability exceeds ``p``, renormalise, and sample one."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # shift right by one so the token that crosses p is still included
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


@torch.inference_mode()
def generate(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    eos_token_id: int,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
    on_token: Optional[Callable[[int], None]] = None,
) -> torch.Tensor:
    """Returns the generated token ids ``[1, T]`` (prompt excluded).

    The first step (prefill) runs image + prompt through the model and fills
    the KV cache; every later step feeds only the newest token.
    """
    kv_cache = KVCache()
    generated: list[torch.Tensor] = []

    for _ in range(max_tokens_to_generate):
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, kv_cache=kv_cache)
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token = next_token.squeeze(0)  # [1]
        generated.append(next_token)
        if on_token is not None:
            on_token(int(next_token.item()))
        if next_token.item() == eos_token_id:
            break

        # after prefill only the new token is fed; image is already in the cache
        input_ids = next_token.unsqueeze(-1)
        pixel_values = None
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device, dtype=attention_mask.dtype)], dim=-1)

    return torch.cat(generated, dim=-1).unsqueeze(0)
