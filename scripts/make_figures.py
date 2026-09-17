"""Generate the README figures into ``assets/``.

* kv_cache_memory.png     - analytical KV-cache footprint (Gemma-2B decoder config)
* autocast_throughput.png - measured tokens/s on this machine, fp32 vs autocast, at three decoder widths
* architecture.png        - PaliGemma data-flow diagram
* detection_decoder.png   - <loc> token decoding demo on the documented example output
* results.json            - the raw numbers behind the charts

Run: python scripts/make_figures.py [--skip-throughput]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ASSETS = os.path.join(ROOT, "assets")

from vlm.configs import paligemma_3b_224, paligemma_small  # noqa: E402
from vlm.detection import parse_detections  # noqa: E402
from vlm.generate import generate  # noqa: E402
from vlm.paligemma import PaliGemmaForConditionalGeneration  # noqa: E402
from vlm.utils import autocast_context, autocast_dtype, get_device  # noqa: E402

# Okabe-Ito colour-blind-safe palette; categorical hues are assigned in fixed order.
BLUE, ORANGE, GREY, INK, MUTED = "#0072B2", "#E69F00", "#BBBBBB", "#222222", "#666666"
SURFACE = "#FFFFFF"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#DDDDDD")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)


def kv_bytes_static(cfg, max_len, elem=2):
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * max_len * cfg.head_dim * elem


def kv_bytes_concat(cfg, seq_len, elem=2, kv_heads=None):
    kv_heads = cfg.num_key_value_heads if kv_heads is None else kv_heads
    return 2 * cfg.num_hidden_layers * kv_heads * seq_len * cfg.head_dim * elem


def fig_kv_cache(results):
    cfg = paligemma_3b_224().text_config
    seq = 256 + 16 + 128  # image tokens + prompt + generated
    rows = [
        ("Static cache\nmax_len = 8192", kv_bytes_static(cfg, 8192)),
        ("Static cache\nmax_len = 640", kv_bytes_static(cfg, 640)),
        ("Concatenation\n(exact length, GQA)", kv_bytes_concat(cfg, seq)),
    ]
    mha = kv_bytes_concat(cfg, seq, kv_heads=cfg.num_attention_heads)
    mb = [b / 2**20 for _, b in rows]
    results["kv_cache_mb"] = {name.replace("\n", " "): round(v, 2) for (name, _), v in zip(rows, mb)}
    results["kv_cache_mb"]["Concatenation, full MHA (8 KV heads)"] = round(mha / 2**20, 2)
    results["kv_concat_vs_static640_saving_pct"] = round(100 * (1 - mb[2] / mb[1]), 1)
    results["kv_concat_vs_static8192_saving_pct"] = round(100 * (1 - mb[2] / mb[0]), 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160, facecolor=SURFACE)
    colors = [GREY, GREY, BLUE]
    bars = ax.barh([r[0] for r in rows], mb, color=colors, height=0.55)
    for bar, v in zip(bars, mb):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f"{v:.1f} MB", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("KV-cache memory (MB, bf16)", color=MUTED, fontsize=9)
    ax.set_xlim(0, max(mb) * 1.18)
    ax.set_title(f"KV-cache footprint, Gemma-2B decoder, {seq}-token sequence", loc="left", fontsize=11, color=INK)
    ax.text(0, -0.28, f"concat vs 640-token static: -{results['kv_concat_vs_static640_saving_pct']}%   |   "
            f"vs 8192-token static: -{results['kv_concat_vs_static8192_saving_pct']}%",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    style(ax)
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "kv_cache_memory.png"))
    plt.close(fig)


def build_small(hidden: int, device):
    """paligemma_small-style 12-layer decoder at a given width (head_dim 128, 4 heads, 1 KV head)."""
    cfg = paligemma_small(vocab_size=4096)
    tc = cfg.text_config
    tc.hidden_size, tc.intermediate_size = hidden, hidden * 4
    cfg.hidden_size = cfg.projection_dim = hidden
    cfg = type(cfg)(vision_config=cfg.vision_config, text_config=tc, image_token_index=cfg.image_token_index,
                    vocab_size=cfg.vocab_size, projection_dim=hidden, hidden_size=hidden, pad_token_id=0)
    return PaliGemmaForConditionalGeneration(cfg).to(device).eval(), cfg


def measure_throughput(widths=(512, 1024, 1536), n_tokens=32, runs=3):
    device = get_device()

    def sync():
        if device.type in ("cuda", "mps"):
            getattr(torch, device.type).synchronize()

    rows = []
    for hidden in widths:
        model, cfg = build_small(hidden, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_img = cfg.vision_config.num_image_tokens
        ids = torch.cat([torch.full((1, n_img), cfg.image_token_index), torch.tensor([[2, 10, 11, 12]])], 1).to(device)
        mask = torch.ones_like(ids)
        pix = torch.randn(1, 3, 224, 224, device=device)

        def run(enabled):
            with autocast_context(device, enabled=enabled):
                generate(model, ids, mask, pix, eos_token_id=-1, max_tokens_to_generate=n_tokens, do_sample=True)
            sync()

        row = {"hidden": hidden, "params_M": round(n_params, 1)}
        half = autocast_dtype(device)
        modes = (("fp32", False, torch.float32), ("autocast", True, torch.float32), ("half weights", False, half))
        for name, enabled, wdtype in modes:
            model.to(wdtype)
            run(enabled)  # warm-up (kernel compilation on MPS/CUDA)
            best = 0.0
            for _ in range(runs):
                t = time.time()
                run(enabled)
                best = max(best, n_tokens / (time.time() - t))
            row[name] = round(best, 1)
        row["autocast_speedup"] = round(row["autocast"] / row["fp32"], 2)
        row["half_speedup"] = round(row["half weights"] / row["fp32"], 2)
        row["weights_MB_fp32"] = round(n_params * 4, 0)
        row["weights_MB_half"] = round(n_params * 2, 0)
        rows.append(row)
        print(row, flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    return device, str(autocast_dtype(device)).replace("torch.", ""), rows


def fig_throughput(results, skip):
    if skip:
        device, dtype, rows = "mps", "float16", results.get("throughput_rows", [])
    else:
        device, dtype, rows = measure_throughput()
    results["throughput"] = {"device": str(device), "autocast_dtype": dtype, "rows": rows,
                             "note": "12-layer decoder + 6-layer SigLIP, random weights, 32 generated tokens, best of 3 after warm-up",
                             "machine": platform.processor() or platform.machine()}
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=160, facecolor=SURFACE)
    x = np.arange(len(rows))
    w = 0.26
    fp32 = [r["fp32"] for r in rows]
    ac = [r["autocast"] for r in rows]
    hw = [r["half weights"] for r in rows]
    b1 = ax.bar(x - w - 0.02, fp32, w, color=GREY, label="fp32 weights")
    b2 = ax.bar(x, ac, w, color=ORANGE, label=f"fp32 weights + autocast ({dtype})")
    b3 = ax.bar(x + w + 0.02, hw, w, color=BLUE, label=f"{dtype} weights")
    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.0f}", ha="center", fontsize=8, color=INK)
    allv = fp32 + ac + hw
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={r['hidden']}" + chr(10) + f"{r['params_M']:.0f}M params" for r in rows])
    ax.set_ylabel("decode throughput (tokens / s)", color=MUTED, fontsize=9)
    ax.set_ylim(0, max(allv) * 1.28)
    ax.set_title(f"Precision modes on {str(device).upper()} vs. decoder width (12 layers, batch 1 decode)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "autocast_throughput.png"))
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160, facecolor=SURFACE)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.2, edgecolor=color, facecolor=color + "18"))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center", fontsize=8.2, color=MUTED, linespacing=1.45)

    def arrow(x0, y0, x1, y1, label=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=MUTED))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.16, label, ha="center", fontsize=8, color=MUTED)

    # vision path (top)
    box(0.2, 2.7, 1.6, 1.5, "Image", "224 x 224 x 3\nbicubic resize\n(x - 0.5) / 0.5", GREY.replace("#BBBBBB", "#888888"))
    box(2.3, 2.7, 2.6, 1.5, "SigLIP ViT", "27 pre-LN layers, 1152-d\n16 heads, patch 14\n-> 256 patch tokens", BLUE)
    box(5.4, 2.7, 1.9, 1.5, "Projector", "Linear 1152 -> 2048\n/ sqrt(2048) scale", BLUE)
    # text path (bottom)
    box(0.2, 0.5, 1.6, 1.5, "Prompt", '"detect cat"\n<image>x256 + <bos>\n+ prompt + \\n', GREY.replace("#BBBBBB", "#888888"))
    box(2.3, 0.5, 2.6, 1.5, "Gemma tokenizer", "257,216 vocab\n1024 <loc> + 128 <seg>\nembedding x sqrt(d)", ORANGE)
    box(5.4, 0.5, 1.9, 1.5, "Token merge", "masked_scatter into\n<image> slots\nbidirectional prefix", ORANGE)
    # decoder + head (right)
    box(7.8, 1.3, 3.0, 2.3, "Gemma decoder", "18 layers (12 in small preset), 2048-d\nGQA: 8 query heads -> 1 KV head\nRoPE, RMSNorm(1+w), GeGLU\nconcat KV-cache", ORANGE)
    ax.text(9.3, 0.62, "tied LM head -> top-p sampling", ha="center", fontsize=8.6, color=INK)
    ax.text(9.3, 0.3, "<loc0591><loc0252><loc0941><loc0784> cat", ha="center", fontsize=8.2, color=BLUE, family="monospace")

    arrow(1.8, 3.45, 2.3, 3.45)
    arrow(4.9, 3.45, 5.4, 3.45)
    arrow(7.3, 3.45, 7.8, 3.0)
    ax.text(7.55, 3.62, "image features", ha="center", fontsize=8, color=MUTED)
    arrow(1.8, 1.25, 2.3, 1.25)
    arrow(4.9, 1.25, 5.4, 1.25)
    arrow(6.35, 2.0, 6.35, 2.7)
    arrow(7.3, 1.25, 7.8, 1.9)
    ax.text(7.55, 1.05, "multimodal tokens", ha="center", fontsize=8, color=MUTED)
    arrow(9.3, 1.3, 9.3, 0.85)
    ax.set_title("PaliGemma from scratch: vision-language fusion", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "architecture.png"))
    plt.close(fig)


def fig_detection_demo():
    """Draws the decoder output for the example string published in the Hugging Face
    PaliGemma blog post on a synthetic scene. This exercises ``parse_detections`` and the
    1024-grid rescaling; it is NOT a real model prediction."""
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (236, 240, 244))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 300, w, h], fill=(203, 189, 160))  # ground
    d.rectangle([0, 0, w, 300], fill=(214, 228, 240))  # sky
    text = "<loc0591><loc0252><loc0941><loc0784> dog"
    dets = parse_detections(text, w, h)
    for det in dets:
        x0, y0, x1, y1 = det.box
        d.ellipse([x0 + 20, y0 + 30, x1 - 20, y1 - 10], fill=(150, 110, 70))  # a stand-in "dog"
        d.rectangle(det.box, outline=(0, 114, 178), width=4)
        d.rectangle([x0, y0 - 22, x0 + 190, y0], fill=(0, 114, 178))
        d.text((x0 + 6, y0 - 19), f"{det.label}  ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})", fill="white")
    d.text((10, 8), "decoder demo: " + text, fill=(60, 60, 60))
    img.save(os.path.join(ASSETS, "detection_decoder.png"))
    return [x.to_dict() for x in dets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-throughput", action="store_true", help="use the numbers recorded in the README instead of measuring")
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    results = {"torch": torch.__version__}
    fig_kv_cache(results)
    fig_throughput(results, args.skip_throughput)
    fig_architecture()
    results["detection_decoder_demo"] = fig_detection_demo()
    with open(os.path.join(ASSETS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
