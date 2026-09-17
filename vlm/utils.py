"""Device selection and SafeTensors checkpoint loading."""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import torch
from safetensors import safe_open

from .paligemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU unless ``preferred`` is given."""
    if preferred not in (None, "auto"):
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """bf16 on CUDA (when supported) and fp16 on MPS; CPU autocast uses bf16."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def autocast_context(device: torch.device, enabled: bool = True):
    """``torch.autocast`` configured for the device (MPS autocast landed in torch 2.5)."""
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=autocast_dtype(device), enabled=True)


def load_config(model_path: str) -> PaliGemmaConfig:
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        return PaliGemmaConfig.from_dict(json.load(f))


def load_safetensors(model_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors shards found in {model_path}")
    tensors: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def load_hf_model(
    model_path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> tuple[PaliGemmaForConditionalGeneration, "AutoTokenizer"]:
    """Load a Hugging Face PaliGemma checkpoint directory into the from-scratch model.

    The directory must contain ``config.json``, tokenizer files and ``*.safetensors``.
    Weights are read on CPU and moved to ``device`` once, so peak memory is
    one copy of the model rather than two.
    """
    from transformers import AutoTokenizer

    device = get_device() if device is None else device
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    config = load_config(model_path)
    model = PaliGemmaForConditionalGeneration(config)

    state_dict = load_safetensors(model_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # lm_head is tied to embed_tokens and is not stored in the checkpoint
    missing = [k for k in missing if k != "language_model.lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"Weight mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    model.tie_weights()

    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device).eval()
    return model, tokenizer
