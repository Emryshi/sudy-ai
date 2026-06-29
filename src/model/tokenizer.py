import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

class SudyTokenizer:
    """
    Morphologically-aware BPE Tokenizer wrapper optimized for Turkish NLP.
    Provides methods to train on raw Turkish text corpora and encode/decode sequences.
    """
    def __init__(self, tokenizer_path: str = None):
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 4
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load(tokenizer_path)
        else:
            # Initialize a new default tokenizer configuration
            vocab = {token: i for i, token in enumerate(self.special_tokens)}
            self.tokenizer = Tokenizer(models.BPE(vocab, [], unk_token=self.unk_token))
            # Pre-tokenizer setup: Byte-level pre-tokenization with Turkish morphological splitter regex
            # Splits words on punctuation and Turkish apostrophe separating proper nouns from suffixes.
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            
        self.fast_tokenizer = None

    def train(self, files: list, vocab_size: int = 65536, min_frequency: int = 2):
        """Train the tokenizer on a list of text files."""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train(files, trainer=trainer)
        self._sync_fast_tokenizer()

    def save(self, path: str):
        """Save the tokenizer to a directory or json file."""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        # Save a config map
        config = {
            "vocab_size": self.tokenizer.get_vocab_size(),
            "special_tokens": self.special_tokens,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "mask_token_id": self.mask_token_id
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def load(self, path: str):
        """Load the tokenizer from a directory."""
        json_path = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
        self.tokenizer = Tokenizer.from_file(json_path)
        self._sync_fast_tokenizer()

    def _sync_fast_tokenizer(self):
        """Syncs tokenizers.Tokenizer to transformers.PreTrainedTokenizerFast."""
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """Encode text to token IDs."""
        if not self.fast_tokenizer:
            self._sync_fast_tokenizer()
        
        ids = self.fast_tokenizer.encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if not self.fast_tokenizer:
            self._sync_fast_tokenizer()
        return self.fast_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def add_new_tokens(self, new_tokens: list) -> int:
        """Add a list of new tokens to the vocabulary. Returns number of added tokens."""
        # Filter tokens that are already in vocabulary
        vocab = self.tokenizer.get_vocab()
        tokens_to_add = [t for t in new_tokens if t and t not in vocab]
        if not tokens_to_add:
            return 0
        added = self.tokenizer.add_tokens(tokens_to_add)
        self._sync_fast_tokenizer()
        return added

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
