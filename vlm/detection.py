"""Decode PaliGemma detection output into bounding boxes.

For the ``detect <object>`` task PaliGemma emits, per object::

    <locYMIN><locXMIN><locYMAX><locXMAX> label ; <loc..>... label2

Each ``<locNNNN>`` is a coordinate quantised to 1024 bins over the *resized*
image, so we rescale to the original resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw

_LOC_RE = re.compile(r"<loc(\d{4})>")
_BOX_RE = re.compile(r"((?:<loc\d{4}>){4})\s*([^;<]*)")


@dataclass
class Detection:
    label: str
    box: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in pixels

    def to_dict(self) -> dict:
        return {"label": self.label, "box": [round(v, 2) for v in self.box]}


def loc_to_coord(loc_value: int, extent: int, num_bins: int = 1024) -> float:
    """Map a quantised bin in [0, num_bins) to a pixel coordinate in [0, extent]."""
    return (loc_value / (num_bins - 1)) * extent


def parse_detections(text: str, image_width: int, image_height: int) -> list[Detection]:
    detections: list[Detection] = []
    for match in _BOX_RE.finditer(text):
        locs = [int(v) for v in _LOC_RE.findall(match.group(1))]
        if len(locs) != 4:
            continue
        y_min, x_min, y_max, x_max = locs
        label = match.group(2).strip() or "object"
        detections.append(
            Detection(
                label=label,
                box=(
                    loc_to_coord(x_min, image_width),
                    loc_to_coord(y_min, image_height),
                    loc_to_coord(x_max, image_width),
                    loc_to_coord(y_max, image_height),
                ),
            )
        )
    return detections


def draw_detections(
    image: Image.Image, detections: list[Detection], color: str = "red", width: int = 3, output_path: Optional[str] = None
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        draw.rectangle(det.box, outline=color, width=width)
        draw.text((det.box[0] + 2, max(det.box[1] - 12, 0)), det.label, fill=color)
    if output_path:
        canvas.save(output_path)
    return canvas
