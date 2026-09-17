"""Measure KV-cache memory: grow-by-concatenation vs. pre-allocated static cache.

A static cache reserves ``2 * L * B * H_kv * S_max * Dh`` elements up front for a
fixed ``max_position_embeddings``. Concatenation only ever holds the tokens seen
so far, so its footprint scales with the actual prefix + generated length.

Run: python scripts/benchmark_kv_cache.py [--max_len 8192] [--gen 128]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.configs import paligemma_3b_224
from vlm.gemma import KVCache


def static_cache_bytes(layers: int, batch: int, kv_heads: int, max_len: int, head_dim: int, dtype: torch.dtype) -> int:
    elem = torch.empty((), dtype=dtype).element_size()
    return 2 * layers * batch * kv_heads * max_len * head_dim * elem


def concat_cache_bytes(layers: int, batch: int, kv_heads: int, prefix: int, gen: int, head_dim: int, dtype: torch.dtype) -> int:
    cache = KVCache()
    k = torch.zeros(batch, kv_heads, prefix, head_dim, dtype=dtype)
    for layer in range(layers):
        cache.update(k, k, layer)
    step = torch.zeros(batch, kv_heads, 1, head_dim, dtype=dtype)
    for _ in range(gen):
        for layer in range(layers):
            cache.update(step, step, layer)
    return cache.memory_bytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=256 + 16, help="image tokens + prompt tokens")
    ap.add_argument("--gen", type=int, default=128, help="tokens generated")
    ap.add_argument("--max_len", type=int, default=None, help="static cache length (default: config max_position_embeddings)")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    cfg = paligemma_3b_224().text_config
    dtype = getattr(torch, args.dtype)
    max_len = args.max_len or cfg.max_position_embeddings

    static = static_cache_bytes(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, max_len, cfg.head_dim, dtype)
    concat = concat_cache_bytes(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, args.prefix, args.gen, cfg.head_dim, dtype)
    mha_concat = concat * (cfg.num_attention_heads // cfg.num_key_value_heads)

    mb = 1024**2
    print(f"Gemma-2B decoder: {cfg.num_hidden_layers} layers, {cfg.num_key_value_heads} KV head(s) of {cfg.head_dim}, dtype={args.dtype}")
    print(f"Sequence: prefix={args.prefix} + generated={args.gen} = {args.prefix + args.gen} tokens")
    print()
    print(f"{'Cache strategy':<48}{'MB':>10}")
    print(f"{'static pre-allocated (max_len=' + str(max_len) + ')':<48}{static / mb:>10.2f}")
    print(f"{'concatenation (exact length)':<48}{concat / mb:>10.2f}")
    print(f"{'concatenation, full MHA (no GQA) for reference':<48}{mha_concat / mb:>10.2f}")
    print()
    print(f"concat vs static: {100 * (1 - concat / static):.1f}% less KV-cache memory")
    print(f"GQA vs MHA (same length): {100 * (1 - concat / mha_concat):.1f}% less KV-cache memory")


if __name__ == "__main__":
    main()
