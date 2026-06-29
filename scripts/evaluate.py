import os
import argparse
import torch
import math
import json
from tqdm import tqdm

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer
from src.training.data_utils import TextCleaner

def calculate_perplexity(model, tokenizer, text_file: str, max_length: int = 128) -> float:
    """Computes Perplexity (PPL) on a raw text corpus validation split."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read and clean text corpus
    with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    clean_text = TextCleaner.clean(text)
    token_ids = tokenizer.encode(clean_text, add_special_tokens=True)

    # Chunk tokens into sequences
    chunks = [token_ids[i:i+max_length] for i in range(0, len(token_ids), max_length) if len(token_ids[i:i+max_length]) > 5]
    if not chunks:
        return float('inf')

    total_loss = 0.0
    total_tokens = 0

    print(f"Calculating Perplexity over {len(chunks)} text chunks...")
    with torch.no_grad():
        for chunk in tqdm(chunks):
            # Sequence
            input_tensor = torch.tensor([chunk], dtype=torch.long, device=device)
            # Targets are shifted to the left by 1
            labels = input_tensor.clone()
            
            outputs = model(input_tensor, labels=labels)
            loss = outputs["loss"].item()
            
            # Weighted by number of tokens in the sequence
            num_tokens = len(chunk)
            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return perplexity

def evaluate_sft_bleu(model, tokenizer, sft_file: str) -> float:
    """Evaluates SFT exact match / overlap score against reference answers."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(sft_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Limit to top 20 pairs for fast validation
    qa_pairs = qa_pairs[:20]
    total_score = 0.0

    print(f"Evaluating generation accuracy over {len(qa_pairs)} SFT samples...")
    for pair in tqdm(qa_pairs):
        prompt = pair["prompt"]
        reference = pair["response"]

        # Run model generation
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = model.generate(input_tensor, max_new_tokens=40, temperature=0.0)

        full_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        if full_text.startswith(prompt_decoded):
            generated = full_text[len(prompt_decoded):].strip()
        else:
            generated = full_text

        # Compute simple word overlap (Jaccard similarity)
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        
        if not ref_words and not gen_words:
            score = 1.0
        elif not ref_words or not gen_words:
            score = 0.0
        else:
            score = len(ref_words.intersection(gen_words)) / len(ref_words.union(gen_words))
        
        total_score += score

    avg_score = total_score / len(qa_pairs) if qa_pairs else 0.0
    return avg_score

def main():
    parser = argparse.ArgumentParser(description="Sudy Model Evaluation Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt weights file")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation text or JSON file")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="pretrain", help="Evaluation mode (pretrain for perplexity, sft for BLEU/Overlap)")
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = SudyTokenizer(args.tokenizer_path)
    
    # Load config yaml
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = SudyConfig(vocab_size=tokenizer.get_vocab_size(), **config_dict)
    
    model = SudyLMHeadModel(config)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}. Running evaluation on untrained weights.")

    if args.mode == "pretrain":
        # Calculate perplexity
        perplexity = calculate_perplexity(model, tokenizer, args.val_data_path)
        print(f"\n======================================")
        print(f"Validation Perplexity: {perplexity:.4f}")
        print(f"======================================")
    else:
        # Calculate SFT lexical score
        score = evaluate_sft_bleu(model, tokenizer, args.val_data_path)
        print(f"\n======================================")
        print(f"Validation Generation Similarity (Jaccard): {100 * score:.2f}%")
        print(f"======================================")

if __name__ == "__main__":
    main()
