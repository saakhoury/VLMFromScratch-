import torch

from vlm.siglip import SiglipAttention, SiglipVisionConfig, SiglipVisionModel


def small_cfg():
    return SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8
    )


def test_output_shape_is_one_token_per_patch():
    cfg = small_cfg()
    model = SiglipVisionModel(cfg)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, cfg.num_patches, cfg.hidden_size)
    assert cfg.num_patches == 16


def test_attention_rows_sum_to_one():
    cfg = small_cfg()
    attn = SiglipAttention(cfg)
    _, weights = attn(torch.randn(1, 5, cfg.hidden_size))
    assert weights.shape == (1, cfg.num_attention_heads, 5, 5)
    assert torch.allclose(weights.sum(-1), torch.ones(1, cfg.num_attention_heads, 5), atol=1e-5)


def test_state_dict_keys_match_hf_layout():
    keys = set(SiglipVisionModel(small_cfg()).state_dict().keys())
    assert "vision_model.embeddings.patch_embedding.weight" in keys
    assert "vision_model.encoder.layers.0.self_attn.q_proj.weight" in keys
    assert "vision_model.encoder.layers.1.mlp.fc2.bias" in keys
    assert "vision_model.post_layernorm.weight" in keys
