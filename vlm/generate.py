"""Autoregressive decoding, cache policy, profiling and nucleus sampling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .gemma import KVCache
from .paligemma import PaliGemmaForConditionalGeneration


@dataclass
class GenerationProfile:
    """Runtime measurements for one batch-1 generation."""

    cache_strategy: str
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: list[float]
    cache_allocated_bytes: int
    cache_used_bytes: int

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + sum(self.decode_seconds)

    @property
    def tokens_per_second(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return self.generated_tokens / self.total_seconds

    @property
    def decode_p50_seconds(self) -> float:
        return _percentile(self.decode_seconds, 0.50)

    @property
    def decode_p95_seconds(self) -> float:
        return _percentile(self.decode_seconds, 0.95)

    def to_dict(self) -> dict:
        mb = 1024**2
        return {
            "cache_strategy": self.cache_strategy,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prefill_ms": round(self.prefill_seconds * 1000, 2),
            "decode_p50_ms": round(self.decode_p50_seconds * 1000, 2),
            "decode_p95_ms": round(self.decode_p95_seconds * 1000, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "kv_cache_allocated_mb": round(self.cache_allocated_bytes / mb, 3),
            "kv_cache_used_mb": round(self.cache_used_bytes / mb, 3),
        }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    """Make wall-clock timings meaningful on asynchronous accelerators."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: sample from the smallest set whose mass reaches p."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Shift right conceptually so the token that crosses p is still included.
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def _make_cache(strategy: str, prompt_tokens: int, max_tokens_to_generate: int) -> KVCache:
    if strategy == "growing":
        return KVCache()
    if strategy == "static":
        # One extra slot is harmless and keeps capacity independent of early EOS.
        return KVCache(max_length=prompt_tokens + max_tokens_to_generate)
    raise ValueError("cache_strategy must be 'growing' or 'static'")


@torch.inference_mode()
def _generate_impl(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    eos_token_id: int,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    on_token: Optional[Callable[[int], None]],
    cache_strategy: str,
    collect_profile: bool,
) -> tuple[torch.Tensor, Optional[GenerationProfile]]:
    if max_tokens_to_generate <= 0:
        raise ValueError("max_tokens_to_generate must be positive")

    prompt_tokens = input_ids.shape[1]
    kv_cache = _make_cache(cache_strategy, prompt_tokens, max_tokens_to_generate)
    generated: list[torch.Tensor] = []
    prefill_seconds = 0.0
    decode_seconds: list[float] = []

    for step_idx in range(max_tokens_to_generate):
        if collect_profile:
            _synchronize(input_ids.device)
            started = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if collect_profile:
            _synchronize(input_ids.device)
            elapsed = time.perf_counter() - started
            if step_idx == 0:
                prefill_seconds = elapsed
            else:
                decode_seconds.append(elapsed)

        next_token = next_token.squeeze(0)  # batch size 1 -> [1]
        generated.append(next_token)

        if on_token is not None:
            on_token(int(next_token.item()))
        if next_token.item() == eos_token_id:
            break

        # After prefill, the image and earlier tokens are already represented in
        # the cache. Only the newest token is evaluated on subsequent steps.
        input_ids = next_token.unsqueeze(-1)
        pixel_values = None
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    device=input_ids.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=-1,
        )

    output_ids = torch.cat(generated, dim=-1).unsqueeze(0)

    if not collect_profile:
        return output_ids, None

    profile = GenerationProfile(
        cache_strategy=kv_cache.strategy,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(generated),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        cache_allocated_bytes=kv_cache.memory_bytes(),
        cache_used_bytes=kv_cache.used_memory_bytes(),
    )
    return output_ids, profile


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
    cache_strategy: str = "growing",
) -> torch.Tensor:
    """Generate token ids while preserving the original simple API."""
    output_ids, _ = _generate_impl(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        eos_token_id=eos_token_id,
        max_tokens_to_generate=max_tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        on_token=on_token,
        cache_strategy=cache_strategy,
        collect_profile=False,
    )
    return output_ids


def generate_profiled(
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
    cache_strategy: str = "growing",
) -> tuple[torch.Tensor, GenerationProfile]:
    """Generate and return stage/cache metrics for the same execution path."""
    output_ids, profile = _generate_impl(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        eos_token_id=eos_token_id,
        max_tokens_to_generate=max_tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        on_token=on_token,
        cache_strategy=cache_strategy,
        collect_profile=True,
    )
    assert profile is not None
    return output_ids, profile
