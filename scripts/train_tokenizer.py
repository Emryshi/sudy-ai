import os
import argparse
from src.model import SudyTokenizer

def main():
    parser = argparse.ArgumentParser(description="Sudy Tokenizer Training Script")
    parser.add_argument("--data_file", type=str, required=True, help="Text file to train tokenizer on")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sudy-tokenizer", help="Directory to save tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size (keep small for test, standard is 65536)")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: {args.data_file} not found.")
        return

    print(f"Training tokenizer on {args.data_file} with vocab size {args.vocab_size}...")
    tokenizer = SudyTokenizer()
    tokenizer.train([args.data_file], vocab_size=args.vocab_size)
    
    tokenizer.save(args.output_dir)
    print(f"Tokenizer saved to {args.output_dir}")

    # Verify loading and sample tokenization
    loaded = SudyTokenizer(args.output_dir)
    sample_text = "türkçe büyük dil modeli için verimli tokenizer test ediliyor."
    encoded = loaded.encode(sample_text)
    decoded = loaded.decode(encoded)
    
    print("\nVerification:")
    print(f"Input:  {sample_text}")
    print(f"Tokens: {encoded}")
    print(f"Decoded:{decoded}")


if __name__ == "__main__":
    main()
