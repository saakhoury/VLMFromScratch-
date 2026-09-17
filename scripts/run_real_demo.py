"""Download real PaliGemma weights and produce caption / VQA / detection results.

Needs ~12 GB disk for the fp32 shards and a GPU with >= 8 GB VRAM (or a Mac with
>= 16 GB unified memory). Runs unchanged in Google Colab (T4 is enough):

    !git clone <this repo> && cd VLMFromScratch- && pip install -r requirements.txt
    !python scripts/run_real_demo.py --repo hehe156/paligemma-3b-pt-224

``hehe156/paligemma-3b-pt-224`` is an ungated mirror of ``google/paligemma-3b-pt-224``
(identical config and weight names, verified against its safetensors index). Pass
``--repo google/paligemma-3b-mix-224`` after accepting the licence for the
instruction-mixed checkpoint, which answers free-form prompts better.

Outputs go to ``assets/real/``: annotated detection images and ``results.md``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vlm.detection import draw_detections, parse_detections  # noqa: E402
from vlm.generate import generate  # noqa: E402
from vlm.processing import PaliGemmaProcessor  # noqa: E402
from vlm.utils import autocast_context, get_device, load_hf_model  # noqa: E402

# Public-domain / CC sample images (COCO val2017 via the COCO site).
SAMPLES = {
    "cats": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "bus": "http://images.cocodataset.org/val2017/000000000139.jpg",
    "skateboard": "http://images.cocodataset.org/val2017/000000000785.jpg",
}
PROMPTS = {
    "cats": ["caption en", "detect cat", "answer en how many cats are there?"],
    "bus": ["caption en", "detect tv ; detect person"],
    "skateboard": ["caption en", "detect person", "answer en what is the person doing?"],
}


def fetch(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hehe156/paligemma-3b-pt-224")
    ap.add_argument("--local_dir", default=os.path.join(ROOT, "weights", "paligemma-3b-pt-224"))
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--max_tokens", type=int, default=48)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.local_dir, "config.json")):
        from huggingface_hub import snapshot_download

        print(f"Downloading {args.repo} -> {args.local_dir}")
        snapshot_download(args.repo, local_dir=args.local_dir, allow_patterns=["*.json", "*.model", "*.safetensors"])

    device = get_device()
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}")
    t0 = time.time()
    model, tok = load_hf_model(args.local_dir, device, dtype=dtype)
    print(f"loaded in {time.time() - t0:.1f}s, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    vc = model.config.vision_config
    proc = PaliGemmaProcessor(tok, vc.num_image_tokens, vc.image_size)

    out_dir = os.path.join(ROOT, "assets", "real")
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# Real-weight results", "", f"Checkpoint: `{args.repo}` | device: `{device}` | dtype: `{args.dtype}`", ""]
    summary = {}
    for name, url in SAMPLES.items():
        image = fetch(url)
        image.save(os.path.join(out_dir, f"{name}.jpg"))
        lines += [f"## {name}", "", f"![{name}](real/{name}.jpg)", ""]
        for prompt in PROMPTS[name]:
            inputs = {k: v.to(device) for k, v in proc(text=[prompt], images=[image]).items()}
            t0 = time.time()
            with autocast_context(device):
                ids = generate(model, inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"],
                               eos_token_id=tok.eos_token_id, max_tokens_to_generate=args.max_tokens, do_sample=False)
            dt = time.time() - t0
            text = tok.decode(ids[0], skip_special_tokens=False).replace("<eos>", "").strip()
            tps = ids.shape[1] / dt
            print(f"[{name}] {prompt!r} -> {text!r}  ({tps:.1f} tok/s)")
            row = f"| `{prompt}` | `{text}` | {tps:.1f} tok/s |"
            if "| Prompt |" not in "\n".join(lines[-3:]):
                lines += ["| Prompt | Output | Speed |", "|---|---|---|"]
            lines.append(row)
            summary.setdefault(name, []).append({"prompt": prompt, "output": text, "tok_s": round(tps, 1)})
            if prompt.startswith("detect"):
                dets = parse_detections(text, image.width, image.height)
                if dets:
                    path = os.path.join(out_dir, f"{name}_{prompt.split()[1]}_boxes.jpg")
                    draw_detections(image, dets, output_path=path)
                    lines += ["", f"![{name} detections](real/{os.path.basename(path)})"]
        lines.append("")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
