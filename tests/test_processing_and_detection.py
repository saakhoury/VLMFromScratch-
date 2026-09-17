import numpy as np
import torch
from PIL import Image

from vlm.detection import draw_detections, parse_detections
from vlm.generate import sample_top_p
from vlm.processing import EXTRA_TOKENS, IMAGE_TOKEN, PaliGemmaProcessor, add_image_tokens_to_prompt, process_images


class FakeTokenizer:
    """Minimal stand-in for a SentencePiece tokenizer: whitespace split + added tokens."""

    bos_token = "<bos>"
    eos_token_id = 1

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}

    def add_special_tokens(self, d):
        for t in d["additional_special_tokens"]:
            self.vocab.setdefault(t, len(self.vocab))

    def add_tokens(self, toks):
        for t in toks:
            self.vocab.setdefault(t, len(self.vocab))

    def convert_tokens_to_ids(self, tok):
        return self.vocab[tok]

    def __call__(self, texts, return_tensors="pt", padding=None, truncation=None):
        ids = []
        for text in texts:
            toks = text.replace("<image>", " <image> ").replace("<bos>", " <bos> ").split()
            ids.append([self.vocab.setdefault(t, len(self.vocab)) for t in toks])
        ids = torch.tensor(ids)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_prompt_layout():
    assert add_image_tokens_to_prompt("caption en", "<bos>", 2, "<image>") == "<image><image><bos>caption en\n"


def test_extra_tokens_count():
    assert len(EXTRA_TOKENS) == 1024 + 128
    assert EXTRA_TOKENS[0] == "<loc0000>" and EXTRA_TOKENS[1023] == "<loc1023>" and EXTRA_TOKENS[-1] == "<seg127>"


def test_process_images_normalises_to_minus_one_one():
    img = Image.fromarray(np.full((50, 70, 3), 255, dtype=np.uint8))
    arr = process_images([img], size=(16, 16))[0]
    assert arr.shape == (3, 16, 16) and arr.dtype == np.float32
    assert np.allclose(arr, 1.0)
    black = process_images([Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))], size=(16, 16))[0]
    assert np.allclose(black, -1.0)


def test_processor_prepends_image_tokens():
    tok = FakeTokenizer()
    proc = PaliGemmaProcessor(tok, num_image_tokens=4, image_size=16)
    img = Image.fromarray((np.random.rand(20, 30, 3) * 255).astype(np.uint8))
    out = proc(text=["detect cat"], images=[img])
    assert out["pixel_values"].shape == (1, 3, 16, 16)
    ids = out["input_ids"][0].tolist()
    assert ids[:4] == [proc.image_token_id] * 4
    assert ids[4] == tok.vocab["<bos>"]
    assert out["attention_mask"].all()


def test_parse_detections_scales_to_image_size():
    text = "<loc0000><loc0000><loc1023><loc1023> cat ; <loc0511><loc0511><loc1023><loc1023> dog"
    dets = parse_detections(text, image_width=200, image_height=100)
    assert [d.label for d in dets] == ["cat", "dog"]
    assert dets[0].box == (0.0, 0.0, 200.0, 100.0)
    x0, y0, x1, y1 = dets[1].box
    assert abs(x0 - 99.9) < 0.1 and abs(y0 - 49.95) < 0.1 and x1 == 200.0 and y1 == 100.0


def test_draw_detections_returns_image():
    img = Image.new("RGB", (64, 64))
    dets = parse_detections("<loc0100><loc0100><loc0900><loc0900> box", 64, 64)
    out = draw_detections(img, dets)
    assert out.size == (64, 64) and np.asarray(out).sum() > 0


def test_top_p_never_samples_outside_nucleus():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    samples = {sample_top_p(probs.clone(), 0.7).item() for _ in range(300)}
    assert samples <= {0, 1}
    assert sample_top_p(probs.clone(), 0.0).item() == 0
