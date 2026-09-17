"""PaliGemma input processor: image -> normalised pixel tensor, text -> token ids.

The prompt is laid out exactly as in the released checkpoints::

    <image> * num_image_tokens  +  <bos>  +  prompt  +  "\\n"

The 1024 ``<locXXXX>`` tokens encode bounding-box coordinates on a 1024-bin grid
and the 128 ``<segXXX>`` tokens encode segmentation-mask codes (see ``detection.py``).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
IMAGE_TOKEN = "<image>"
NUM_LOC_TOKENS = 1024
NUM_SEG_TOKENS = 128
LOC_TOKENS = [f"<loc{i:04d}>" for i in range(NUM_LOC_TOKENS)]
SEG_TOKENS = [f"<seg{i:03d}>" for i in range(NUM_SEG_TOKENS)]
EXTRA_TOKENS = LOC_TOKENS + SEG_TOKENS


def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str) -> str:
    # Trailing newline is part of the PaliGemma prefix format and is required.
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(image: Image.Image, size: tuple[int, int], resample: Image.Resampling = Image.Resampling.BICUBIC) -> Image.Image:
    height, width = size
    return image.resize((width, height), resample=resample)


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    return (image * scale).astype(dtype)


def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std


def process_images(
    images: Sequence[Image.Image],
    size: tuple[int, int],
    resample: Image.Resampling = Image.Resampling.BICUBIC,
    rescale_factor: float = 1 / 255.0,
    image_mean: Optional[Iterable[float]] = None,
    image_std: Optional[Iterable[float]] = None,
) -> list[np.ndarray]:
    """resize -> [0,1] -> (x-mean)/std -> CHW. Returns one float32 array per image."""
    image_mean = IMAGENET_STANDARD_MEAN if image_mean is None else image_mean
    image_std = IMAGENET_STANDARD_STD if image_std is None else image_std
    out = []
    for image in images:
        image = image.convert("RGB")
        image = resize(image, size=size, resample=resample)
        arr = np.array(image)
        arr = rescale(arr, scale=rescale_factor)
        arr = normalize(arr, mean=image_mean, std=image_std)
        out.append(arr.transpose(2, 0, 1))  # HWC -> CHW
    return out


class PaliGemmaProcessor:
    """Pairs a (Gemma) tokenizer with SigLIP image preprocessing.

    ``tokenizer`` needs ``add_special_tokens``, ``add_tokens``, ``convert_tokens_to_ids``,
    ``__call__`` and ``bos_token`` - the Hugging Face ``AutoTokenizer`` for any Gemma
    checkpoint works, but so does any duck-typed object (see tests).
    """

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int) -> None:
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        # We add BOS/EOS ourselves in the prompt template.
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Sequence[str],
        images: Sequence[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        if len(images) != 1 or len(text) != 1:
            raise ValueError(f"Received {len(images)} images for {len(text)} prompts; only batch size 1 is supported.")

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        pixel_values = torch.tensor(np.stack(pixel_values, axis=0))  # [B, C, H, W]

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=IMAGE_TOKEN,
            )
            for prompt in text
        ]
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding=padding, truncation=truncation)
        return {"pixel_values": pixel_values, **inputs}
