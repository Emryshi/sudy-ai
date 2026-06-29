import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer
from src.training.data_utils import SudyDataset

def get_lr_scheduler(optimizer, warmup_steps, total_steps, base_lr):
    """Cosine learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Initialize tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = SudyTokenizer(args.tokenizer_path)

    # Update config with vocab size
    config_dict["vocab_size"] = tokenizer.get_vocab_size()
    config_dict["pad_token_id"] = tokenizer.pad_token_id
    config_dict["bos_token_id"] = tokenizer.bos_token_id
    config_dict["eos_token_id"] = tokenizer.eos_token_id

    # Initialize model config and model
    config = SudyConfig(**config_dict)
    print("Initializing Sudy model...")
    model = SudyLMHeadModel(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Load pretraining data
    # In a real setup, this would load massive corpora. Here we read local raw files or generate synthetic data.
    texts = []
    if args.data_path and os.path.exists(args.data_path):
        print(f"Reading training data from {args.data_path}...")
        if args.data_path.endswith(".csv"):
            import csv
            with open(args.data_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    text_line = " ".join([cell.strip() for cell in row if cell.strip()])
                    if text_line:
                        texts.append(text_line)
        else:
            with open(args.data_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        print("Data path not found. Generating synthetic Turkish sentences for pretraining...")
        texts = [
            "türkçe büyük dil modeli geliştirme projesi tüm hızıyla devam ediyor.",
            "yapay zeka alanındaki yenilikler hayatımızı kolaylaştırmaya devam ediyor.",
            "sudy mimarisi daha verimli dikkat mekanizmaları sunmaktadır.",
            "morfolojik tokenizer yapısı türkçe eklerini çok iyi analiz edebilir.",
            "veri ön işleme aşamasında metin temizliği ve tekilleştirme kritik öneme sahiptir.",
            "büyük dil modelleri parametre sayısına bağlı olarak daha yetenekli hale gelir.",
            "makine öğrenmesi modelleri veri kalitesi arttıkça daha iyi sonuçlar üretir.",
            "verimli antrenman süreçleri donanım kaynaklarını en iyi şekilde kullanmayı gerektirir.",
        ] * 50

    dataset = SudyDataset(texts, tokenizer, max_length=args.max_length, is_sft=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, args.lr)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"Starting pretraining for {args.epochs} epochs. Total steps: {total_steps}")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=None, # will construct internally
                    labels=labels
                )
                loss = outputs["loss"]
            
            # Scale loss and backpropagate
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "aux_loss": f"{outputs['aux_loss'].item():.4f}"
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        config.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    config.save_pretrained(args.output_dir)
    print(f"Saved final pretrain model to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudy Pretrain Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--data_path", type=str, default="", help="Path to raw text data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sudy-pretrain", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU/CPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")

    args = parser.parse_args()
    train(args)
