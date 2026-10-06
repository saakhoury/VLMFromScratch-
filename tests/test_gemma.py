import torch

from vlm.gemma import (
    GemmaAttention,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    KVCache,
    apply_rotary_pos_emb,
    repeat_kv,
)


def cfg(**kw):
    base = dict(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )
    base.update(kw)
    return GemmaConfig(**base)


def test_rmsnorm_zero_weight_gives_unit_rms():
    norm = GemmaRMSNorm(16)
    x = torch.randn(4, 16) * 5
    rms = norm(x).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-4)


def test_rope_preserves_norm_and_is_relative():
    rope = GemmaRotaryEmbedding(8)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    cos, sin = rope(q, torch.arange(4)[None])
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_r.norm(dim=-1), atol=1e-5)
    # shifting all positions by a constant leaves q.k dot products unchanged
    cos2, sin2 = rope(q, torch.arange(4)[None] + 10)
    q_s, k_s = apply_rotary_pos_emb(q, k, cos2, sin2)
    assert torch.allclose(q_r @ k_r.transpose(-1, -2), q_s @ k_s.transpose(-1, -2), atol=1e-4)


def test_repeat_kv_shape_and_content():
    kv = torch.randn(2, 2, 5, 8)
    out = repeat_kv(kv, 3)
    assert out.shape == (2, 6, 5, 8)
    assert torch.equal(out[:, 0], kv[:, 0]) and torch.equal(out[:, 2], kv[:, 0]) and torch.equal(out[:, 3], kv[:, 1])


def test_gqa_matches_mha_when_kv_heads_are_copied():
    mha = GemmaAttention(cfg(num_key_value_heads=4), 0)
    gqa = GemmaAttention(cfg(num_key_value_heads=2), 0)
    gqa.q_proj.load_state_dict(mha.q_proj.state_dict())
    gqa.o_proj.load_state_dict(mha.o_proj.state_dict())
    # make MHA's 4 KV heads equal to GQA's 2 heads repeated (head h -> group h // 2)
    for name in ("k_proj", "v_proj"):
        w = getattr(gqa, name).weight.data.view(2, 8, 32)
        getattr(mha, name).weight.data = w[[0, 0, 1, 1]].reshape(32, 32)
    x = torch.randn(1, 6, 32)
    pos = torch.arange(6)[None]
    assert torch.allclose(mha(x, None, pos)[0], gqa(x, None, pos)[0], atol=1e-5)


def test_kv_cache_grows_by_concatenation():
    cache = KVCache()
    k = torch.randn(1, 2, 5, 8)
    cache.update(k, k, 0)
    assert cache.num_items() == 5
    k1 = torch.randn(1, 2, 1, 8)
    keys, _ = cache.update(k1, k1, 0)
    assert cache.num_items() == 6 and torch.equal(keys[:, :, -1], k1[:, :, 0])
    assert cache.memory_bytes() == 2 * 1 * 2 * 6 * 8 * 4


def test_cached_decoding_matches_full_recompute():
    model = GemmaForCausalLM(cfg()).eval()
    ids = torch.randint(0, 100, (1, 7))
    emb = model.get_input_embeddings()
    causal = torch.triu(torch.full((7, 7), float("-inf")), diagonal=1)[None, None]
    with torch.no_grad():
        full = model(emb(ids), causal, torch.arange(7)[None])["logits"]
        cache = KVCache()
        prefix_mask = causal[:, :, :6, :6]
        model(emb(ids[:, :6]), prefix_mask, torch.arange(6)[None], cache)
        step = model(emb(ids[:, 6:]), None, torch.tensor([[6]]), cache)["logits"]
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-4)


def test_lm_head_is_tied_to_embeddings():
    model = GemmaForCausalLM(cfg())
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" in model.state_dict()


def test_static_kv_cache_preallocates_and_tracks_used_tokens():
    cache = KVCache(max_length=10)
    k = torch.randn(1, 2, 5, 8)
    keys, values = cache.update(k, k, 0)

    assert cache.strategy == "static"
    assert cache.num_items() == 5
    assert keys.shape[-2] == 5 and values.shape[-2] == 5
    assert cache.memory_bytes() == 2 * 1 * 2 * 10 * 8 * 4
    assert cache.used_memory_bytes() == 2 * 1 * 2 * 5 * 8 * 4

    k1 = torch.randn(1, 2, 1, 8)
    keys, _ = cache.update(k1, k1, 0)
    assert cache.num_items() == 6
    assert torch.equal(keys[:, :, -1], k1[:, :, 0])
