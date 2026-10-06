"""Benchmark the actual batch-1 runtime path without downloading 3B weights.

This uses the same PaliGemma module tree and generation loop as real inference,
but initializes a reduced preset with random weights so cache/device experiments
are fast and reproducible on a laptop.

Run:
    python scripts/benchmark_runtime.py
    python scripts/benchmark_runtime.py --preset tiny --tokens 8
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.configs import paligemma_small, paligemma_tiny  # noqa: E402
from vlm.generate import generate_profiled  # noqa: E402
from vlm.paligemma import PaliGemmaForConditionalGeneration  # noqa: E402
from vlm.utils import autocast_context, get_device, model_parameter_bytes  # noqa: E402


def build_inputs(cfg, device: torch.device):
    n_img = cfg.vision_config.num_image_tokens
    ids = torch.cat(
        [
            torch.full((1, n_img), cfg.image_token_index),
            torch.tensor([[2, 10, 11, 12]]),
        ],
        dim=1,
    ).to(device)
    mask = torch.ones_like(ids)
    pixels = torch.randn(
        1,
        3,
        cfg.vision_config.image_size,
        cfg.vision_config.image_size,
        device=device,
    )
    return ids, mask, pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["small", "tiny"], default="small")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--autocast", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = get_device(args.device)
    cfg = (
        paligemma_small(vocab_size=4096)
        if args.preset == "small"
        else paligemma_tiny()
    )
    model = PaliGemmaForConditionalGeneration(cfg).to(device).eval()
    ids, mask, pixels = build_inputs(cfg, device)

    print(
        f"device={device} preset={args.preset} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
        f"weights={model_parameter_bytes(model) / 1024**2:.1f}MB"
    )
    print()
    print(
        f"{'cache':<10}{'prefill ms':>12}{'decode p50':>13}"
        f"{'decode p95':>13}{'tok/s':>10}{'alloc MB':>11}{'used MB':>10}"
    )

    outputs = {}
    for strategy in ("growing", "static"):
        with autocast_context(device, enabled=args.autocast):
            output, profile = generate_profiled(
                model,
                ids,
                mask,
                pixels,
                eos_token_id=-1,
                max_tokens_to_generate=args.tokens,
                do_sample=False,
                cache_strategy=strategy,
            )
        outputs[strategy] = output
        summary = profile.to_dict()
        print(
            f"{strategy:<10}"
            f"{summary['prefill_ms']:>12.2f}"
            f"{summary['decode_p50_ms']:>13.2f}"
            f"{summary['decode_p95_ms']:>13.2f}"
            f"{summary['tokens_per_second']:>10.1f}"
            f"{summary['kv_cache_allocated_mb']:>11.2f}"
            f"{summary['kv_cache_used_mb']:>10.2f}"
        )

    if not torch.equal(outputs["growing"], outputs["static"]):
        raise RuntimeError("Cache strategies changed greedy model output")

    print()
    print("semantic check: growing output == static output")


if __name__ == "__main__":
    main()
