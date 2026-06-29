import pytest
import os
import tempfile
from src.model import SudyTokenizer

def test_tokenizer_encoding():
    tokenizer = SudyTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = os.path.join(tmpdir, "train.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("türkçe dil yapısı test ediliyor.\n")
        tokenizer.train([train_file], vocab_size=100, min_frequency=1)

    text = "türkçe dil yapısı test ediliyor."
    
    encoded = tokenizer.encode(text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id
    assert text in decoded.lower()

def test_tokenizer_train_and_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy training data
        train_file = os.path.join(tmpdir, "train.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("türkçe dil yapısı eklemeli bir dildir.\n")
            f.write("yapay zeka alanında büyük ilerlemeler yapıldı.\n")
            f.write("bu bir test cümlesidir.\n")

        tokenizer = SudyTokenizer()
        tokenizer.train([train_file], vocab_size=100, min_frequency=1)
        
        save_path = os.path.join(tmpdir, "tokenizer_model")
        tokenizer.save(save_path)
        
        # Load and test
        loaded = SudyTokenizer(save_path)
        assert loaded.get_vocab_size() > 5 # special tokens + letters
        
        text = "yapay zeka test"
        ids = loaded.encode(text, add_special_tokens=False)
        decoded = loaded.decode(ids)
        assert decoded.strip() == text
