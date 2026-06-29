import os
import argparse
import sys
import json

# Ensure stdout encodes correctly on CP1254 Windows shells
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='ignore')

def download_and_process(repo_id: str, split: str = "train", is_sft: bool = True, limit: int = 5000):
    print(f"HuggingFace dataset download requested: repo_id='{repo_id}', split='{split}', mode={'SFT' if is_sft else 'Pretrain'}")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library is not installed. Please install it using: pip install datasets")
        sys.exit(1)
        
    try:
        print(f"Connecting to HuggingFace Hub and loading dataset: {repo_id}...")
        ds = load_dataset(repo_id, split=split)
        print(f"Successfully loaded split '{split}' with {len(ds)} rows.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset from HuggingFace: {e}")
        sys.exit(1)
        
    # Enforce rows limit to prevent disk/RAM overflow
    ds_subset = ds.select(range(min(len(ds), limit)))
    print(f"Processing first {len(ds_subset)} rows...")
    
    safe_name = repo_id.replace("/", "_").replace("-", "_")
    os.makedirs("data/processed", exist_ok=True)
    
    if is_sft:
        # Detect SFT columns
        cols = ds_subset.column_names
        prompt_keys = ["prompt", "instruction", "giriş", "soru", "madde", "question", "text"]
        response_keys = ["response", "output", "çıkış", "cevap", "anlam", "answer", "target"]
        
        prompt_col = None
        response_col = None
        
        for c in cols:
            if c.lower() in prompt_keys:
                prompt_col = c
            if c.lower() in response_keys:
                response_col = c
                
        # If no distinct cols found, fallback to first two cols
        if not prompt_col or not response_col:
            if len(cols) >= 2:
                prompt_col = cols[0]
                response_col = cols[1]
            elif len(cols) == 1:
                prompt_col = cols[0]
                response_col = cols[0]
            else:
                print("[ERROR] Dataset has no columns to parse.")
                sys.exit(1)
                
        print(f"Mapping prompt column: '{prompt_col}', response column: '{response_col}'")
        
        sft_data = []
        for row in ds_subset:
            p_val = str(row[prompt_col]).strip()
            r_val = str(row[response_col]).strip() if response_col != prompt_col else ""
            
            if p_val:
                sft_data.append({
                    "prompt": p_val,
                    "response": r_val
                })
                
        output_file = f"data/processed/hf_{safe_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=4)
        print(f"[SUCCESS] Dataset successfully saved as JSON SFT: {output_file} ({len(sft_data)} rows)")
        
    else:
        # Pretraining text dump
        text_lines = []
        for row in ds_subset:
            # Concatenate all cell text
            row_str = " ".join([str(val).strip() for val in row.values() if val is not None])
            if row_str.strip():
                text_lines.append(row_str.strip())
                
        output_file = f"data/processed/hf_{safe_name}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in text_lines:
                f.write(line + "\n")
        print(f"[SUCCESS] Dataset successfully saved as Pretraining text: {output_file} ({len(text_lines)} lines)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudy HuggingFace Dataset Downloader Utility")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repository ID (e.g. eryk/turkish-squad)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, test, validation, etc.)")
    parser.add_argument("--type", type=str, default="sft", choices=["sft", "pretrain"], help="Output format type (sft or pretrain)")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum rows to fetch and process")
    
    args = parser.parse_args()
    download_and_process(
        repo_id=args.repo_id,
        split=args.split,
        is_sft=(args.type == "sft"),
        limit=args.limit
    )
