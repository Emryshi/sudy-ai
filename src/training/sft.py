import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer
from src.training.data_utils import SudyDataset

def train_sft(args):
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Initialize tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = SudyTokenizer(args.tokenizer_path)

    config_dict["vocab_size"] = tokenizer.get_vocab_size()
    config_dict["pad_token_id"] = tokenizer.pad_token_id
    config_dict["bos_token_id"] = tokenizer.bos_token_id
    config_dict["eos_token_id"] = tokenizer.eos_token_id

    # Initialize model config and model
    config = SudyConfig(**config_dict)
    print("Initializing Sudy model for SFT...")
    model = SudyLMHeadModel(config)

    # Load pre-trained weights if provided
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        print(f"Loading pretrained weights from {args.pretrain_checkpoint}...")
        model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location="cpu"))
    else:
        print("Warning: Starting SFT from scratch (no pretrain checkpoint provided).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Load SFT dataset
    prompts = []
    responses = []
    
    if args.data_path and os.path.exists(args.data_path):
        print(f"Reading SFT data from {args.data_path}...")
        if args.data_path.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(args.data_path)
                prompt_cols = ["prompt", "instruction", "giriş", "soru", "madde"]
                response_cols = ["response", "output", "çıkış", "cevap", "anlam"]
                
                found_prompt = None
                found_response = None
                for c in df.columns:
                    if str(c).lower() in prompt_cols:
                        found_prompt = c
                    if str(c).lower() in response_cols:
                        found_response = c
                        
                if found_prompt and found_response:
                    for _, row in df.iterrows():
                        prompts.append(str(row[found_prompt]))
                        responses.append(str(row[found_response]))
                else:
                    for _, row in df.iterrows():
                        prompts.append(str(row.iloc[0]))
                        responses.append(str(row.iloc[1]))
            except Exception as e:
                import csv
                with open(args.data_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            prompts.append(row[0])
                            responses.append(row[1])
        else:
            import json
            with open(args.data_path, "r", encoding="utf-8") as f:
                if args.data_path.endswith(".jsonl"):
                    for line in f:
                        data = json.loads(line)
                        prompts.append(data["prompt"])
                        responses.append(data["response"])
                else:
                    data_list = json.load(f)
                    for item in data_list:
                        prompts.append(item["prompt"])
                        responses.append(item["response"])
    else:
        print("SFT data path not found. Creating synthetic SFT pairs (Turkish instruction/response)...")
        # Generates basic Turkish QA examples for training validation
        qa_pairs = [
            ("türkiye'nin başkenti neresidir?", "türkiye'nin başkenti ankara'dır."),
            ("yapay zeka nedir?", "yapay zeka, insan zekasını taklit eden bilgisayar sistemleridir."),
            ("en hızlı koşan hayvan hangisidir?", "dünyanın en hızlı koşan kara hayvanı çitadır."),
            ("sudy modelinin amacı nedir?", "sudy, verimli türkçe büyük dil modeli geliştirmeyi amaçlayan bir projedir."),
            ("güneş sistemindeki en büyük gezegen hangisidir?", "güneş sistemindeki en büyük gezegen jüpiter'dir."),
            ("istatistiksel analiz neden önemlidir?", "istatistiksel analiz, verilerden anlamlı çıkarımlar yapmak için önemlidir."),
            ("python nedir?", "python, okunabilirliği yüksek olan genel amaçlı bir programlama dilidir."),
            ("suyun formülü nedir?", "suyun kimyasal formülü h2o'dur.")
        ] * 40
        prompts = [q for q, a in qa_pairs]
        responses = [a for q, a in qa_pairs]

    dataset = SudyDataset(prompts, tokenizer, max_length=args.max_length, is_sft=True, targets=responses)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"Starting SFT training for {args.epochs} epochs.")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=labels
                )
                loss = outputs["loss"]

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"SFT Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-sft-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        config.save_pretrained(checkpoint_dir)
        print(f"Saved SFT checkpoint to {checkpoint_dir}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    config.save_pretrained(args.output_dir)
    print(f"Saved final SFT model to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudy SFT Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--pretrain_checkpoint", type=str, default="", help="Path to pretrained model.pt")
    parser.add_argument("--data_path", type=str, default="", help="Path to SFT data file")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sudy-sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU/CPU")
    parser.add_argument("--lr", type=float, default=5e-5, help="SFT learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")

    args = parser.parse_args()
    train_sft(args)
