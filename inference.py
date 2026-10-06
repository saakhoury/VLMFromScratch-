"""Run PaliGemma on an image + prompt with explicit runtime controls."""

from __future__ import annotations

import json
import time
from typing import Optional

import fire
import torch
from PIL import Image

from vlm.detection import draw_detections, parse_detections
from vlm.generate import generate, generate_profiled
from vlm.processing import PaliGemmaProcessor
from vlm.utils import (
    autocast_context,
    get_device,
    load_hf_model,
    model_parameter_bytes,
    resolve_weight_dtype,
)


def move_inputs_to_device(model_inputs: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in model_inputs.items()}


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: Optional[str] = None,
    weight_dtype: str = "auto",
    autocast: bool = True,
    cache_strategy: str = "growing",
    profile: bool = False,
    save_detections_to: Optional[str] = None,
) -> None:
    device_ = get_device(device)
    weight_dtype_ = resolve_weight_dtype(device_, weight_dtype)

    print(
        f"Device: {device_} | weight dtype: {weight_dtype_} | "
        f"autocast: {autocast} | cache: {cache_strategy}"
    )

    print("Loading model...")
    started = time.perf_counter()
    model, tokenizer = load_hf_model(model_path, device_, dtype=weight_dtype_)
    load_seconds = time.perf_counter() - started
    weight_mb = model_parameter_bytes(model) / 1024**2
    print(f"Loaded in {load_seconds:.1f}s | parameter memory: {weight_mb:.1f} MB")

    vision_cfg = model.config.vision_config
    processor = PaliGemmaProcessor(
        tokenizer,
        vision_cfg.num_image_tokens,
        vision_cfg.image_size,
    )

    image = Image.open(image_file_path)
    model_inputs = move_inputs_to_device(
        processor(text=[prompt], images=[image]),
        device_,
    )

    print(f"Prompt: {prompt}\nOutput: ", end="", flush=True)

    with autocast_context(device_, enabled=autocast):
        if profile:
            out_ids, runtime = generate_profiled(
                model,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs["pixel_values"],
                eos_token_id=tokenizer.eos_token_id,
                max_tokens_to_generate=max_tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                cache_strategy=cache_strategy,
            )
        else:
            started = time.perf_counter()
            out_ids = generate(
                model,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs["pixel_values"],
                eos_token_id=tokenizer.eos_token_id,
                max_tokens_to_generate=max_tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                cache_strategy=cache_strategy,
            )
            elapsed = time.perf_counter() - started
            runtime = None

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    print(decoded)

    if runtime is not None:
        print("Runtime profile:")
        print(json.dumps(runtime.to_dict(), indent=2))
    else:
        print(
            f"[{out_ids.shape[1]} tokens in {elapsed:.2f}s, "
            f"{out_ids.shape[1] / elapsed:.1f} tok/s]"
        )

    if prompt.strip().lower().startswith("detect"):
        detections = parse_detections(decoded, image.width, image.height)
        print("Detections:", json.dumps([item.to_dict() for item in detections], indent=2))
        if save_detections_to and detections:
            draw_detections(image, detections, output_path=save_detections_to)
            print(f"Saved annotated image to {save_detections_to}")


if __name__ == "__main__":
    fire.Fire(main)
