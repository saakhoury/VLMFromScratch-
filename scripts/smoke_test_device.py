"""End-to-end smoke test of the 12-layer ``paligemma_small`` preset with random
weights on the auto-selected device (CUDA / MPS / CPU) under autocast.

Run: python scripts/smoke_test_device.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.configs import paligemma_small  # noqa: E402
from vlm.generate import generate  # noqa: E402
from vlm.paligemma import PaliGemmaForConditionalGeneration  # noqa: E402
from vlm.utils import autocast_context, autocast_dtype, get_device  # noqa: E402


def main() -> None:
    device = get_device()
    cfg = paligemma_small(vocab_size=4096)  # small vocab keeps the random model light
    model = PaliGemmaForConditionalGeneration(cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} autocast dtype={autocast_dtype(device)} params={n_params / 1e6:.1f}M")

    n_img = cfg.vision_config.num_image_tokens
    ids = torch.cat([torch.full((1, n_img), cfg.image_token_index), torch.tensor([[2, 10, 11, 12]])], 1).to(device)
    mask = torch.ones_like(ids)
    pix = torch.randn(1, 3, cfg.vision_config.image_size, cfg.vision_config.image_size, device=device)

    t0 = time.time()
    with autocast_context(device):
        out = generate(model, ids, mask, pix, eos_token_id=-1, max_tokens_to_generate=16, do_sample=True, top_p=0.9)
    if device.type in ("cuda", "mps"):
        getattr(torch, device.type).synchronize()
    dt = time.time() - t0
    print(f"generated {out.shape[1]} tokens in {dt:.2f}s ({out.shape[1] / dt:.1f} tok/s) -> {out[0].tolist()}")
    assert out.shape == (1, 16) and torch.isfinite(out.float()).all()
    print("OK")


if __name__ == "__main__":
    main()
