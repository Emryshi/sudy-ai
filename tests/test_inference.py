import pytest
from fastapi.testclient import TestClient
import torch
import os

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer
from src.inference.api import app

def test_generation():
    config = SudyConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        d_latent=16,
        num_experts=2,
        top_k=1,
        num_mtp_heads=0
    )
    model = SudyLMHeadModel(config)
    tokenizer = SudyTokenizer()
    
    prompt = "başkent neresi"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    
    generated = model.generate(input_tensor, max_new_tokens=5, temperature=0.0)
    assert len(generated[0]) > len(prompt_ids)

def test_api_health():
    client = TestClient(app)
    # Trigger startup event to initialize dummy model/tokenizer
    with client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_api_generate():
    client = TestClient(app)
    with client:
        # Test batch generate endpoint
        response = client.post("/generate", json={
            "prompt": "türkiye'nin başkenti neresidir?",
            "max_new_tokens": 10,
            "temperature": 0.0,
            "stream": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data
        assert "tokens_generated" in data
        assert data["tokens_generated"] > 0

def test_api_generate_speculative():
    client = TestClient(app)
    with client:
        # Test batch generate endpoint with speculative decoding enabled
        response = client.post("/generate", json={
            "prompt": "yapay zeka nedir?",
            "max_new_tokens": 10,
            "temperature": 0.0,
            "stream": False,
            "use_speculative": True,
            "mtp_threshold": 0.5
        })
        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data
