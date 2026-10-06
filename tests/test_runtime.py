import torch

from vlm.configs import paligemma_tiny
from vlm.generate import generate, generate_profiled
from vlm.paligemma import PaliGemmaForConditionalGeneration
from vlm.utils import resolve_weight_dtype


def make_inputs():
    cfg = paligemma_tiny()
    model = PaliGemmaForConditionalGeneration(cfg).eval()
    n_img = cfg.vision_config.num_image_tokens
    ids = torch.cat(
        [
            torch.full((1, n_img), cfg.image_token_index),
            torch.tensor([[2, 10, 11, 12]]),
        ],
        dim=1,
    )
    mask = torch.ones_like(ids)
    pixels = torch.randn(
        1,
        3,
        cfg.vision_config.image_size,
        cfg.vision_config.image_size,
    )
    return model, ids, mask, pixels


def test_growing_and_static_cache_produce_same_greedy_output():
    torch.manual_seed(7)
    model, ids, mask, pixels = make_inputs()

    growing = generate(
        model,
        ids,
        mask,
        pixels,
        eos_token_id=-1,
        max_tokens_to_generate=4,
        do_sample=False,
        cache_strategy="growing",
    )
    static = generate(
        model,
        ids,
        mask,
        pixels,
        eos_token_id=-1,
        max_tokens_to_generate=4,
        do_sample=False,
        cache_strategy="static",
    )

    assert torch.equal(growing, static)


def test_profile_reports_prefill_decode_and_cache_memory():
    torch.manual_seed(11)
    model, ids, mask, pixels = make_inputs()

    output, profile = generate_profiled(
        model,
        ids,
        mask,
        pixels,
        eos_token_id=-1,
        max_tokens_to_generate=4,
        do_sample=False,
        cache_strategy="static",
    )

    assert output.shape == (1, 4)
    assert profile.generated_tokens == 4
    assert profile.prompt_tokens == ids.shape[1]
    assert profile.prefill_seconds > 0
    assert len(profile.decode_seconds) == 3
    assert profile.cache_allocated_bytes >= profile.cache_used_bytes > 0
    assert profile.to_dict()["cache_strategy"] == "static"


def test_weight_dtype_policy_is_explicit():
    assert resolve_weight_dtype(torch.device("cpu"), "auto") == torch.float32
    assert resolve_weight_dtype(torch.device("mps"), "auto") == torch.float16
    assert resolve_weight_dtype(torch.device("cpu"), "bf16") == torch.bfloat16
