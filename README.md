# VLMFromScratch

A from-scratch PyTorch reproduction of **PaliGemma** (SigLIP vision encoder + Gemma decoder) that loads the
released `google/paligemma-3b-pt-224` SafeTensors and runs image captioning, VQA and object detection
on CUDA, Apple MPS or CPU.

Everything in `vlm/` is hand-written: contrastive-pretrained ViT (SigLIP), multi-head attention, grouped-query
attention, rotary position embeddings, RMSNorm, KV-cache, weight tying, multimodal token merging and top-p sampling.

## Architecture

```
 image 224x224 ──► SigLIP ViT (27 layers, 1152-d, 16 heads, patch 14) ──► 256 patch tokens
                                                                             │
                                                            Linear projector 1152 ─► 2048
                                                                             │
 "<image>"×256 + <bos> + prompt + "\n" ──► token embeddings ──► masked_scatter merge ─┐
                                                                                        ▼
                    Gemma decoder (18 layers, 2048-d, 8 query heads / 1 KV head, RoPE, GeGLU, RMSNorm)
                                                                                        │
                                                       tied LM head ─► top-p sampling ─► "<loc0123><loc0456>... cat"
```

| Component | File | Notes |
|---|---|---|
| SigLIP encoder | `vlm/siglip.py` | Conv patchify, learned positions, pre-LN blocks, tanh-GELU MLP, post-LN |
| Gemma decoder | `vlm/gemma.py` | `1+w` RMSNorm, RoPE, GQA + `repeat_kv`, concat KV-cache, GeGLU, `sqrt(d)` embed scale, tied `lm_head` |
| Fusion | `vlm/paligemma.py` | Projector + `<image>`-slot merging, bidirectional prefix mask, position ids |
| Processor | `vlm/processing.py` | Bicubic resize, `[-1,1]` normalisation, `<image>`/`<loc>`/`<seg>` tokens, prompt layout |
| Detection | `vlm/detection.py` | Parses `<locYYYY>` 1024-bin boxes back to pixels, draws them |
| Generation | `vlm/generate.py` | KV-cached prefill + decode loop, nucleus (top-p) sampling, greedy |
| Runtime | `vlm/utils.py` | CUDA → MPS → CPU auto-placement, `torch.autocast` (bf16 / fp16), SafeTensors loading |
| Presets | `vlm/configs.py` | `paligemma_3b_224` (HF weights), `paligemma_small` (12-layer decoder), `paligemma_tiny` (tests) |

### Grouped-query attention & KV-cache

Gemma-2B uses 8 query heads that share **one** key/value head. The cache stores only the KV head and
`repeat_kv` broadcasts it to the query groups at attention time. The cache itself is grown with
`torch.cat` along the sequence axis, so it is always exactly `prefix + generated` tokens long instead
of a pre-allocated `max_position_embeddings` block.

`python scripts/benchmark_kv_cache.py` (bf16, 272-token prefix + 128 generated):

| Cache strategy | MB |
|---|---|
| Static pre-allocated, `max_len=640` | 11.25 |
| Static pre-allocated, `max_len=8192` (model max) | 144.00 |
| **Concatenation (exact length)** | **7.03** |
| Concatenation with full MHA instead of GQA | 56.25 |

Against a static cache sized to a typical 640-token budget the concatenated cache uses **~40 % less memory**
(37.5 % here); against the model's full context it is 95 % less, and GQA alone cuts the cache 8× versus MHA.

### Mixed-precision inference (Apple M-series, `paligemma_small`, 52.8 M params, random weights)

| Mode | tok/s |
|---|---|
| fp32, no autocast | 10.8 |
| **fp16 autocast** | **29.1** |

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # 24 tests, < 1 s on CPU
python scripts/smoke_test_device.py   # end-to-end on your GPU / MPS with autocast
```

### Download weights

The checkpoint is gated; accept the licence on Hugging Face first.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download google/paligemma-3b-pt-224 --local-dir weights/paligemma-3b-pt-224
```

### Run inference

```bash
python inference.py \
  --model_path weights/paligemma-3b-pt-224 \
  --prompt "detect cat" \
  --image_file_path cat.jpg \
  --max_tokens_to_generate 64 \
  --save_detections_to out.jpg        # draws parsed <loc> boxes
```

Other prompts: `"caption en"`, `"answer en what colour is the car?"`, `"segment dog"` (returns `<seg>` codes).
Flags: `--do_sample True --temperature 0.8 --top_p 0.9` for nucleus sampling, `--device cpu`,
`--autocast False`. `launch_inference.sh` wraps the same call with environment variables.

### Use as a library

```python
from vlm.utils import load_hf_model, get_device, autocast_context
from vlm.processing import PaliGemmaProcessor
from vlm.generate import generate
from PIL import Image

device = get_device()
model, tok = load_hf_model("weights/paligemma-3b-pt-224", device)
proc = PaliGemmaProcessor(tok, model.config.vision_config.num_image_tokens, model.config.vision_config.image_size)
inputs = {k: v.to(device) for k, v in proc(text=["caption en"], images=[Image.open("cat.jpg")]).items()}
with autocast_context(device):
    ids = generate(model, inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"],
                   eos_token_id=tok.eos_token_id, max_tokens_to_generate=32)
print(tok.decode(ids[0]))
```

## Tests

`tests/` checks the pieces that are easy to get subtly wrong:

- RoPE preserves vector norms and gives position-shift-invariant dot products
- GQA reproduces MHA exactly when KV heads are duplicated
- Cached single-token decoding matches a full recompute
- Image features land in the `<image>` slots at the right scale; text tokens are untouched
- Top-p never samples outside the nucleus; `<loc>` parsing rescales to the source image
- State-dict key names match the Hugging Face checkpoint layout

## References

- Beyer et al., *PaliGemma: A versatile 3B VLM for transfer* (2024)
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (2023)
- Gemma Team, *Gemma: Open Models Based on Gemini Research and Technology* (2024)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (top-p sampling, 2020)
- Hugging Face `transformers` SigLIP / Gemma / PaliGemma implementations (used for checkpoint layout and processor format)
