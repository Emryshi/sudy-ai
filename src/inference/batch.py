import os
import argparse
import torch
import json
from tqdm import tqdm

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer

def run_batch_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Batch Inference active. Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = SudyTokenizer(args.tokenizer_path)

    # Load model
    print(f"Loading model config and weights from {args.model_path}...")
    config = SudyConfig.from_pretrained(args.model_path)
    config.vocab_size = tokenizer.get_vocab_size()
    model = SudyLMHeadModel(config)
    
    weights_path = os.path.join(args.model_path, "model.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Warning: model.pt file not found in model_path. Running generation using initialized parameters.")
    
    model.to(device)
    model.eval()

    # Read inputs
    prompts = []
    if os.path.exists(args.input_file):
        with open(args.input_file, "r", encoding="utf-8") as f:
            if args.input_file.endswith(".jsonl"):
                for line in f:
                    data = json.loads(line)
                    prompts.append(data.get("prompt", ""))
            else:
                prompts = [line.strip() for line in f if line.strip()]
    else:
        print("Input file not found. Generating default sample prompts...")
        prompts = [
            "türkiye'nin başkenti neresidir?",
            "yapay zeka hakkında bilgi ver.",
            "gezegenler nasıl oluşur?",
            "en hızlı koşan hayvan hangisidir?"
        ]

    print(f"Loaded {len(prompts)} prompts. Processing...")

    results = []
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating"):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            generated_ids = model.generate(
                input_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id
            )
            
            full_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            if full_text.startswith(prompt_decoded):
                generated_text = full_text[len(prompt_decoded):].strip()
            else:
                generated_text = full_text

            results.append({
                "prompt": prompt,
                "generated_response": generated_text
            })

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        if args.output_file.endswith(".jsonl"):
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        else:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Batch generation completed. Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudy Batch Inference tool")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--input_file", type=str, default="", help="Path to input text/jsonl file")
    parser.add_argument("--output_file", type=str, default="./outputs/results.json", help="Path to save results")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top_k filtering parameter")

    args = parser.parse_args()
    run_batch_inference(args)
