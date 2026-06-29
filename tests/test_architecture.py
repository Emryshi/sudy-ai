import pytest
import torch
from src.model import SudyConfig, SudyLMHeadModel

def test_config():
    config = SudyConfig(hidden_size=256, num_attention_heads=4)
    assert config.hidden_size == 256
    assert config.num_attention_heads == 4

def test_model_forward():
    config = SudyConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_latent=32,
        num_experts=4,
        top_k=1,
        num_mtp_heads=2,
        max_position_embeddings=128
    )
    model = SudyLMHeadModel(config)
    model.eval()

    # Batch of 2 sequences, each of length 10
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 1000, (2, 10))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert "aux_loss" in outputs
    assert "mtp_losses" in outputs

    # Logits shape should be [batch, seq_len, vocab_size]
    assert outputs["logits"].shape == (2, 10, 1000)
    # 2 MTP prediction heads should each produce loss
    assert len(outputs["mtp_losses"]) == 2

def test_ntk_rope():
    from src.model.architecture import RotaryEmbedding
    # Context length 16, dimension 32
    rope = RotaryEmbedding(dim=32, max_position_embeddings=16)
    
    # 1. Normal length (within bounds)
    cos_1, sin_1 = rope(torch.randn(1, 1, 10, 32), seq_len=10)
    assert cos_1.shape == (10, 32)
    
    # 2. Long context (exceeds max_position_embeddings)
    cos_2, sin_2 = rope(torch.randn(1, 1, 32, 32), seq_len=32)
    assert cos_2.shape == (32, 32)

def test_generate_speculative():
    config = SudyConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        d_latent=16,
        num_experts=2,
        top_k=1,
        num_mtp_heads=1,
        max_position_embeddings=32
    )
    model = SudyLMHeadModel(config)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 5))
    
    with torch.no_grad():
        output = model.generate_speculative(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
            mtp_threshold=0.0  # Force MTP acceptance to test speculative paths
        )
    
    # Check that it generated tokens successfully
    assert output.shape[1] > 5
