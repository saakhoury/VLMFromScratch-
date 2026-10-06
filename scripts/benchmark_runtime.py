"""Benchmark cache/runtime policy with warm-up, paired trials and optional sweeps.

The benchmark deliberately uses the same multimodal prefill/decode path as real
inference, but runs a reduced preset with random weights so it is reproducible
on a laptop without downloading the 3B checkpoint.

Methodology:
- warm up *both* cache strategies before measuring
- alternate strategy order on every trial to reduce "second run wins" bias
- compare identical deterministic greedy outputs on every paired trial
- report medians rather than a single lucky run
- optionally sweep generation budgets and save a machine-readable JSON artifact

Examples:
    python scripts/benchmark_runtime.py
    python scripts/benchmark_runtime.py --trials 7 --warmup 2
    python scripts/benchmark_runtime.py --sweep 8,16,32,64 --json-out runtime.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.configs import paligemma_small, paligemma_tiny  # noqa: E402
from vlm.generate import GenerationProfile, generate_profiled  # noqa: E402
from vlm.paligemma import PaliGemmaForConditionalGeneration  # noqa: E402
from vlm.utils import autocast_context, get_device, model_parameter_bytes  # noqa: E402


STRATEGIES = ("growing", "static")


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


def run_once(
    model,
    ids,
    mask,
    pixels,
    device: torch.device,
    autocast: bool,
    strategy: str,
    tokens: int,
):
    with autocast_context(device, enabled=autocast):
        return generate_profiled(
            model,
            ids,
            mask,
            pixels,
            eos_token_id=-1,
            max_tokens_to_generate=tokens,
            do_sample=False,
            cache_strategy=strategy,
        )


def warm_up(
    model,
    ids,
    mask,
    pixels,
    device: torch.device,
    autocast: bool,
    tokens: int,
    rounds: int,
) -> None:
    # Warm both policies every round. Reverse the order every other round so
    # neither policy is systematically the final warm-up before measurement.
    for round_idx in range(rounds):
        order = STRATEGIES if round_idx % 2 == 0 else tuple(reversed(STRATEGIES))
        for strategy in order:
            run_once(
                model,
                ids,
                mask,
                pixels,
                device,
                autocast,
                strategy,
                tokens,
            )


def summarize(profiles: list[GenerationProfile]) -> dict:
    def median(values):
        return statistics.median(values)

    mb = 1024**2
    decode_samples = [
        step
        for profile in profiles
        for step in profile.decode_seconds
    ]
    return {
        "runs": len(profiles),
        "prefill_ms": median([p.prefill_seconds for p in profiles]) * 1000,
        "decode_p50_ms": (
            statistics.median(decode_samples) * 1000 if decode_samples else 0.0
        ),
        "decode_p95_ms": (
            _percentile(decode_samples, 0.95) * 1000 if decode_samples else 0.0
        ),
        "tokens_per_second": median([p.tokens_per_second for p in profiles]),
        "kv_cache_allocated_mb": median(
            [p.cache_allocated_bytes / mb for p in profiles]
        ),
        "kv_cache_used_mb": median(
            [p.cache_used_bytes / mb for p in profiles]
        ),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def paired_benchmark(
    model,
    ids,
    mask,
    pixels,
    device: torch.device,
    autocast: bool,
    tokens: int,
    trials: int,
) -> dict:
    profiles: dict[str, list[GenerationProfile]] = {
        strategy: [] for strategy in STRATEGIES
    }

    for trial_idx in range(trials):
        # Alternate order to reduce bias from caches, allocator state and device
        # warm-up. Every trial is still a direct paired comparison.
        order = STRATEGIES if trial_idx % 2 == 0 else tuple(reversed(STRATEGIES))
        outputs = {}

        for strategy in order:
            output, profile = run_once(
                model,
                ids,
                mask,
                pixels,
                device,
                autocast,
                strategy,
                tokens,
            )
            outputs[strategy] = output
            profiles[strategy].append(profile)

        if not torch.equal(outputs["growing"], outputs["static"]):
            raise RuntimeError(
                f"Cache strategies changed greedy output on trial {trial_idx + 1}"
            )

    summary = {
        strategy: summarize(profiles[strategy])
        for strategy in STRATEGIES
    }
    summary["semantic_equivalence"] = True
    summary["static_vs_growing_throughput_ratio"] = (
        summary["static"]["tokens_per_second"]
        / summary["growing"]["tokens_per_second"]
        if summary["growing"]["tokens_per_second"] > 0
        else 0.0
    )
    return summary


def print_table(tokens: int, summary: dict) -> None:
    print(f"generation budget={tokens} tokens")
    print(
        f"{'cache':<10}{'prefill ms':>12}{'decode p50':>13}"
        f"{'decode p95':>13}{'tok/s':>10}{'alloc MB':>11}{'used MB':>10}"
    )
    for strategy in STRATEGIES:
        row = summary[strategy]
        print(
            f"{strategy:<10}"
            f"{row['prefill_ms']:>12.2f}"
            f"{row['decode_p50_ms']:>13.2f}"
            f"{row['decode_p95_ms']:>13.2f}"
            f"{row['tokens_per_second']:>10.1f}"
            f"{row['kv_cache_allocated_mb']:>11.2f}"
            f"{row['kv_cache_used_mb']:>10.2f}"
        )
    print(
        "semantic check: growing output == static output "
        f"across {summary['growing']['runs']} paired trials"
    )
    print(
        "static/growing median throughput ratio: "
        f"{summary['static_vs_growing_throughput_ratio']:.2f}x"
    )


def parse_sweep(value: str | None, default_tokens: int) -> list[int]:
    if value is None:
        return [default_tokens]

    tokens = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not tokens or any(token <= 0 for token in tokens):
        raise ValueError("--sweep must contain positive comma-separated token counts")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["small", "tiny"], default="small")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--sweep",
        default=None,
        help="Optional comma-separated generation budgets, e.g. 8,16,32,64",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for a reproducible benchmark artifact",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    torch.manual_seed(7)
    device = get_device(args.device)
    cfg = (
        paligemma_small(vocab_size=4096)
        if args.preset == "small"
        else paligemma_tiny()
    )
    model = PaliGemmaForConditionalGeneration(cfg).to(device).eval()
    ids, mask, pixels = build_inputs(cfg, device)
    budgets = parse_sweep(args.sweep, args.tokens)

    metadata = {
        "device": str(device),
        "preset": args.preset,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "autocast": args.autocast,
        "warmup_rounds": args.warmup,
        "paired_trials": args.trials,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weight_bytes": model_parameter_bytes(model),
        "prompt_tokens": ids.shape[1],
    }

    print(
        f"device={device} preset={args.preset} "
        f"params={metadata['parameters'] / 1e6:.1f}M "
        f"weights={metadata['weight_bytes'] / 1024**2:.1f}MB "
        f"warmup={args.warmup} trials={args.trials}"
    )

    all_results = {}
    for budget in budgets:
        print()
        warm_up(
            model,
            ids,
            mask,
            pixels,
            device,
            args.autocast,
            budget,
            args.warmup,
        )
        result = paired_benchmark(
            model,
            ids,
            mask,
            pixels,
            device,
            args.autocast,
            budget,
            args.trials,
        )
        all_results[str(budget)] = result
        print_table(budget, result)

    artifact = {
        "metadata": metadata,
        "results": all_results,
    }

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print()
        print(f"wrote benchmark artifact: {path}")


if __name__ == "__main__":
    main()
