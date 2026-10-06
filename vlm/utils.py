"""Device selection, precision policy and SafeTensors checkpoint loading."""

from __future__ import annotations

import glob
import json
import os
from contextlib import contextmanager
from typing import Iterator, Optional

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
    """Choose the compute autocast dtype for the selected accelerator."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def resolve_weight_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    """Choose the stored model-weight dtype separately from compute autocast.

    ``auto`` favors fitting the model on accelerators: bf16 on capable CUDA,
    fp16 on MPS / older CUDA, and fp32 on CPU. The explicit choices make the
    memory/performance policy visible to callers instead of hiding it in casts.
    """
    requested = requested.lower()
    if requested == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested not in choices:
        raise ValueError(
            "weight_dtype must be one of auto, float32/fp32, float16/fp16, bfloat16/bf16"
        )
    return choices[requested]


def autocast_context(device: torch.device, enabled: bool = True):
    """``torch.autocast`` configured for the device."""
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype(device),
        enabled=True,
    )


def model_parameter_bytes(model: torch.nn.Module) -> int:
    """Physical bytes occupied by parameters, counting tied storage once."""
    seen = set()
    total = 0
    for parameter in model.parameters():
        pointer = parameter.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += parameter.numel() * parameter.element_size()
    return total


def load_config(model_path: str) -> PaliGemmaConfig:
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as file:
        return PaliGemmaConfig.from_dict(json.load(file))


def _safetensor_files(model_path: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors shards found in {model_path}")
    return files


def load_safetensors(model_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load all shards into one dict.

    Kept as a convenience helper; ``load_hf_model`` intentionally does not
    use this path because a full materialized state dict raises peak host memory.
    """
    tensors: dict[str, torch.Tensor] = {}
    for path in _safetensor_files(model_path):
        with safe_open(path, framework="pt", device=device) as file:
            for key in file.keys():
                tensors[key] = file.get_tensor(key)
    return tensors


@contextmanager
def _temporary_default_dtype(dtype: Optional[torch.dtype]) -> Iterator[None]:
    """Initialize model parameters directly in the requested floating dtype."""
    if dtype is None:
        yield
        return

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_hf_model(
    model_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[PaliGemmaForConditionalGeneration, "AutoTokenizer"]:
    """Load a released PaliGemma directory into the from-scratch runtime.

    Checkpoint shards are applied one at a time rather than first collecting the
    entire state dict in memory. Parameters are also initialized directly in the
    requested dtype. Peak host memory is therefore roughly the model plus one
    SafeTensors shard, instead of the model plus every checkpoint shard.

    The loader verifies that every expected persistent weight is present and
    rejects unexpected keys. ``lm_head`` is intentionally absent from the
    released checkpoint because it is tied to the token embedding.
    """
    from transformers import AutoTokenizer

    device = get_device() if device is None else device
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    config = load_config(model_path)
    with _temporary_default_dtype(dtype):
        model = PaliGemmaForConditionalGeneration(config)

    expected_keys = set(model.state_dict().keys())
    expected_keys.discard("language_model.lm_head.weight")
    loaded_keys = set()

    for path in _safetensor_files(model_path):
        shard = {}
        with safe_open(path, framework="pt", device="cpu") as file:
            for key in file.keys():
                if key not in expected_keys:
                    raise RuntimeError(f"Unexpected checkpoint key: {key}")
                shard[key] = file.get_tensor(key)

        _, unexpected = model.load_state_dict(shard, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
        loaded_keys.update(shard.keys())
        del shard

    missing = sorted(expected_keys - loaded_keys)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing[:5]}")

    model.tie_weights()
    model = model.to(device).eval()
    return model, tokenizer
