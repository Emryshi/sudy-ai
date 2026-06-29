import re
import unicodedata
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Set

class TextCleaner:
    """Utility to clean and normalize Turkish text."""
    @staticmethod
    def lowercase_tr(text: str) -> str:
        """Turkish-aware lowercasing (correctly handles I/ı and İ/i)."""
        text = text.replace('I', 'ı').replace('İ', 'i')
        return text.lower()

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        # Remove HTML tags and scripts
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]*>', ' ', text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        text = TextCleaner.lowercase_tr(text)
        
        return text.strip()


class MinHashLSH:
    """
    MinHash Locality Sensitive Hashing (LSH) for document deduplication.
    Identifies documents with high Jaccard similarity (e.g. > 0.7).
    """
    def __init__(self, num_hashes: int = 128, threshold: float = 0.7):
        self.num_hashes = num_hashes
        self.threshold = threshold
        # Generate random coefficients for hash functions: h(x) = (a * x + b) % c
        # Use a large prime number for modulo
        self.prime = 4294967311
        import random
        random.seed(42)
        self.a = [random.randint(1, self.prime - 1) for _ in range(num_hashes)]
        self.b = [random.randint(0, self.prime - 1) for _ in range(num_hashes)]
        
        self.seen_hashes = set()

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Generate word k-grams from text."""
        words = text.split()
        if len(words) <= k:
            return {text}
        return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

    def _hash_shingle(self, shingle: str) -> int:
        """Simple deterministic hash for a string shingle."""
        # Use Python's built-in hash but make it positive and within limits
        return abs(hash(shingle)) % 2**32

    def compute_signature(self, text: str) -> List[int]:
        """Compute the MinHash signature for a document."""
        shingles = self._get_shingles(text)
        shingle_hashes = [self._hash_shingle(s) for s in shingles]
        
        signature = []
        for i in range(self.num_hashes):
            min_val = float('inf')
            for h in shingle_hashes:
                # Hash function: (a*x + b) % prime
                hash_val = (self.a[i] * h + self.b[i]) % self.prime
                if hash_val < min_val:
                    min_val = hash_val
            signature.append(min_val if min_val != float('inf') else 0)
        return signature

    def is_duplicate(self, text: str) -> bool:
        """
        Check if document is a duplicate by comparing signature bands.
        LSH divides signature into bands to find candidate duplicates.
        """
        if len(text.split()) < 10:
            return False
            
        sig = self.compute_signature(text)
        # Check signature footprint (we can use signature tuple directly for exact/near match)
        # Or divide into bands. For simplicity in memory, we check if signature is highly similar to seen.
        sig_tuple = tuple(sig)
        
        # Approximate check: if any of the signature bands match.
        # Let's check with bands of size 4.
        band_size = 4
        num_bands = self.num_hashes // band_size
        
        is_dup = False
        for i in range(num_bands):
            band = sig_tuple[i*band_size : (i+1)*band_size]
            band_key = (i, band)
            if band_key in self.seen_hashes:
                is_dup = True
            else:
                self.seen_hashes.add(band_key)
                
        return is_dup


class SudyDataset(Dataset):
    """PyTorch Dataset for Pretraining and SFT."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, is_sft: bool = False, targets: List[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_sft = is_sft
        self.examples = []

        if is_sft and targets is not None:
            # SFT Mode: text is the prompt/instruction, target is the response
            for p, r in zip(texts, targets):
                p_clean = TextCleaner.clean(p)
                r_clean = TextCleaner.clean(r)
                
                # Encode prompt and response
                p_ids = self.tokenizer.encode(p_clean, add_special_tokens=True)
                r_ids = self.tokenizer.encode(r_clean, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                
                input_ids = p_ids + r_ids
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                
                # Mask out prompt tokens in labels (loss is only calculated on the response)
                labels = [-100] * len(p_ids) + r_ids[-(len(input_ids) - len(p_ids)):]
                # Pad to max_length
                padding_len = max_length - len(input_ids)
                if padding_len > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
                    labels = labels + [-100] * padding_len
                    
                attention_mask = [1] * (max_length - padding_len) + [0] * padding_len
                
                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                })
        else:
            # Pretraining Mode: standard autoregressive prediction of entire sequence
            for text in texts:
                text_clean = TextCleaner.clean(text)
                ids = self.tokenizer.encode(text_clean, add_special_tokens=True)
                
                if len(ids) > max_length:
                    # Slice into multiple windows if sequence is long
                    for i in range(0, len(ids), max_length):
                        chunk = ids[i : i + max_length]
                        if len(chunk) < 10:  # skip too small chunks
                            continue
                        padding_len = max_length - len(chunk)
                        input_ids = chunk + [self.tokenizer.pad_token_id] * padding_len
                        labels = chunk + [-100] * padding_len
                        attention_mask = [1] * len(chunk) + [0] * padding_len
                        
                        self.examples.append({
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long)
                        })
                else:
                    padding_len = max_length - len(ids)
                    input_ids = ids + [self.tokenizer.pad_token_id] * padding_len
                    labels = ids + [-100] * padding_len
                    attention_mask = [1] * len(ids) + [0] * padding_len
                    
                    self.examples.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long)
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
