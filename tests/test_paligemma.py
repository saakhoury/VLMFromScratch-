import torch

from vlm.configs import paligemma_3b_224, paligemma_small, paligemma_tiny
from vlm.gemma import KVCache
from vlm.generate import generate
from vlm.paligemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration


def make_inputs(cfg, prompt_len=4):
    n_img = cfg.vision_config.num_image_tokens
    ids = torch.cat([
        torch.full((1, n_img), cfg.image_token_index),
        torch.tensor([[2]]),
        torch.randint(3, 100, (1, prompt_len)),
    ], dim=1)
    pix = torch.randn(1, 3, cfg.vision_config.image_size, cfg.vision_config.image_size)
    return ids, torch.ones_like(ids), pix


def test_image_tokens_are_replaced_by_projected_features():
    cfg = paligemma_tiny()
    model = PaliGemmaForConditionalGeneration(cfg).eval()
    ids, mask, pix = make_inputs(cfg)
    with torch.no_grad():
        feats = model.encode_images(pix, torch.float32)
        emb = model.language_model.get_input_embeddings()(ids)
        merged, attn, pos = model._merge_input_ids_with_image_features(feats, emb, ids, mask, None)
    n_img = cfg.vision_config.num_image_tokens
    assert torch.allclose(merged[:, :n_img] * cfg.hidden_size**0.5, feats, atol=1e-5)
    assert torch.equal(merged[:, n_img:], emb[:, n_img:])
    assert attn.shape == (1, 1, ids.shape[1], ids.shape[1]) and (attn == 0).all()
    assert torch.equal(pos[0], torch.arange(1, ids.shape[1] + 1))


def test_forward_and_cached_step_shapes():
    cfg = paligemma_tiny()
    model = PaliGemmaForConditionalGeneration(cfg).eval()
    ids, mask, pix = make_inputs(cfg)
    cache = KVCache()
    with torch.no_grad():
        out = model(ids, pix, mask, cache)
        assert out["logits"].shape == (1, ids.shape[1], cfg.vocab_size)
        assert cache.num_items() == ids.shape[1]
        nxt = out["logits"][:, -1].argmax(-1, keepdim=True)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype)], -1)
        out2 = model(nxt, None, mask, cache)
    assert out2["logits"].shape == (1, 1, cfg.vocab_size)
    assert cache.num_items() == ids.shape[1] + 1


def test_generate_greedy_is_deterministic_and_stops_at_eos():
    cfg = paligemma_tiny()
    model = PaliGemmaForConditionalGeneration(cfg).eval()
    ids, mask, pix = make_inputs(cfg)
    a = generate(model, ids, mask, pix, eos_token_id=1, max_tokens_to_generate=6, do_sample=False)
    b = generate(model, ids, mask, pix, eos_token_id=1, max_tokens_to_generate=6, do_sample=False)
    assert a.shape[0] == 1 and 1 <= a.shape[1] <= 6
    assert torch.equal(a, b)
    if a.shape[1] < 6:
        assert a[0, -1].item() == 1


def test_generate_sampling_runs():
    cfg = paligemma_tiny()
    model = PaliGemmaForConditionalGeneration(cfg).eval()
    ids, mask, pix = make_inputs(cfg)
    out = generate(model, ids, mask, pix, eos_token_id=-1, max_tokens_to_generate=5, do_sample=True, top_p=0.9)
    assert out.shape == (1, 5)


def test_config_from_hf_dict_roundtrip():
    d = {
        "image_token_index": 257152, "projection_dim": 2048, "hidden_size": 2048, "vocab_size": 257216,
        "text_config": {"hidden_size": 2048, "intermediate_size": 16384, "num_attention_heads": 8,
                        "num_hidden_layers": 18, "num_key_value_heads": 1, "vocab_size": 257216, "model_type": "gemma"},
        "vision_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                          "num_hidden_layers": 27, "patch_size": 14, "projection_dim": 2048, "model_type": "siglip_vision_model"},
    }
    cfg = PaliGemmaConfig.from_dict(d)
    assert cfg.vision_config.num_image_tokens == 256
    assert cfg.text_config.num_key_value_heads == 1


def test_presets():
    big = paligemma_3b_224()
    assert big.text_config.num_hidden_layers == 18 and big.vision_config.num_image_tokens == 256
    small = paligemma_small()
    assert small.text_config.num_hidden_layers == 12
    assert small.text_config.num_attention_heads % small.text_config.num_key_value_heads == 0


def test_3b_state_dict_keys_match_checkpoint_layout():
    # build only the key names cheaply via a tiny config with the same module tree
    keys = set(PaliGemmaForConditionalGeneration(paligemma_tiny()).state_dict().keys())
    assert "vision_tower.vision_model.embeddings.patch_embedding.weight" in keys
    assert "multi_modal_projector.linear.weight" in keys
    assert "language_model.model.embed_tokens.weight" in keys
    assert "language_model.model.layers.0.self_attn.k_proj.weight" in keys
    assert "language_model.model.layers.0.mlp.gate_proj.weight" in keys
    assert "language_model.model.layers.1.post_attention_layernorm.weight" in keys
    assert "language_model.model.norm.weight" in keys
