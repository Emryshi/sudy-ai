import os
import argparse
from src.training.data_utils import TextCleaner, MinHashLSH

def preprocess(input_path: str, output_path: str, deduplicate: bool = True):
    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} does not exist.")
        return

    print(f"Starting preprocessing of {input_path}...")
    cleaner = TextCleaner()
    lsh = MinHashLSH() if deduplicate else None
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    unique_count = 0
    duplicate_count = 0
    total_count = 0
    
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            total_count += 1
            cleaned = cleaner.clean(line)
            if not cleaned:
                continue
                
            if deduplicate:
                if lsh.is_duplicate(cleaned):
                    duplicate_count += 1
                    continue
            
            outfile.write(cleaned + "\n")
            unique_count += 1

    print("Preprocessing completed!")
    print(f"Total lines read: {total_count}")
    print(f"Unique lines saved: {unique_count}")
    print(f"Duplicates filtered: {duplicate_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudy Preprocessing Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to raw text file")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed text file")
    parser.add_argument("--deduplicate", action="store_true", help="Enable MinHash LSH deduplication")

    args = parser.parse_args()
    preprocess(args.input, args.output, args.deduplicate)
