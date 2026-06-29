import pytest
import torch
from src.training.data_utils import TextCleaner, MinHashLSH, SudyDataset
from src.model import SudyTokenizer, SudyConfig, SudyLMHeadModel

def test_text_cleaner():
    raw_text = "<script>alert(1)</script> <b>İstanbul'daki</b> tren GARINDAN kalkan...  "
    cleaned = TextCleaner.clean(raw_text)
    # lowercase_tr should lowercase 'İstanbul' to 'istanbul' (preserving correct dotless/dotted i's)
    assert "istanbul'daki" in cleaned
    assert "garından" in cleaned
    assert "<script>" not in cleaned
    assert "<b>" not in cleaned

def test_minhash_lsh():
    lsh = MinHashLSH(num_hashes=32, threshold=0.7)
    doc1 = "yapay zeka ve derin öğrenme alanındaki son makaleleri okumayı seviyorum."
    doc2 = "yapay zeka ve derin öğrenme alanındaki son makaleleri okumayı seviyorum." # Exact duplicate
    doc3 = "tamamen farklı başka bir konu hakkında yazı yazıyoruz."
    
    assert not lsh.is_duplicate(doc1)
    assert lsh.is_duplicate(doc2)
    assert not lsh.is_duplicate(doc3)

def test_dataset():
    tokenizer = SudyTokenizer()
    texts = [
        "bu birinci ön eğitim cümlesidir.",
        "ikinci ön eğitim cümlesi de buradadır."
    ]
    dataset = SudyDataset(texts, tokenizer, max_length=32, is_sft=False)
    assert len(dataset) > 0
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 32
