"""Run PaliGemma on an image + prompt.

Example::

    python inference.py --model_path weights/paligemma-3b-pt-224 \
        --prompt "detect cat" --image_file_path cat.jpg --max_tokens_to_generate 64

Uses CUDA or Apple MPS automatically with autocast mixed precision; pass
``--device cpu`` or ``--no-autocast`` to override.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import fire
import torch
from PIL import Image

from vlm.detection import draw_detections, parse_detections
from vlm.generate import generate
from vlm.processing import PaliGemmaProcessor
from vlm.utils import autocast_context, get_device, load_hf_model


def move_inputs_to_device(model_inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in model_inputs.items()}


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: Optional[str] = None,
    autocast: bool = True,
    save_detections_to: Optional[str] = None,
) -> None:
    device_ = get_device(device)
    print(f"Device: {device_} | autocast: {autocast}")

    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_hf_model(model_path, device_)
    print(f"Loaded in {time.time() - t0:.1f}s")

    vision_cfg = model.config.vision_config
    processor = PaliGemmaProcessor(tokenizer, vision_cfg.num_image_tokens, vision_cfg.image_size)

    image = Image.open(image_file_path)
    model_inputs = move_inputs_to_device(processor(text=[prompt], images=[image]), device_)

    print(f"Prompt: {prompt}\nOutput: ", end="", flush=True)
    t0 = time.time()
    with autocast_context(device_, enabled=autocast):
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
        )
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    elapsed = time.time() - t0
    print(decoded)
    print(f"[{out_ids.shape[1]} tokens in {elapsed:.2f}s, {out_ids.shape[1] / elapsed:.1f} tok/s]")

    if prompt.strip().lower().startswith("detect"):
        detections = parse_detections(decoded, image.width, image.height)
        print("Detections:", json.dumps([d.to_dict() for d in detections], indent=2))
        if save_detections_to and detections:
            draw_detections(image, detections, output_path=save_detections_to)
            print(f"Saved annotated image to {save_detections_to}")


if __name__ == "__main__":
    fire.Fire(main)
