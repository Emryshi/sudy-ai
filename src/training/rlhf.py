import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer

class TurkishRewardManager:
    """
    Reward function manager for Turkish LLM alignment.
    Evaluates responses based on length, formatting, and semantic matching rules.
    """
    def __init__(self):
        pass

    def evaluate(self, prompt: str, response: str) -> float:
        reward = 0.0
        
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # 1. Format reward (Turkish sentence capitalization and end of sentence full stop)
        if response and response[0].isupper():
            reward += 0.2
        if response and response[-1] in ['.', '!', '?']:
            reward += 0.2

        # 2. Length penalty/reward (discourage empty or overly long gibberish)
        word_count = len(response.split())
        if word_count < 3:
            reward -= 0.5
        elif 3 <= word_count <= 25:
            reward += 0.3
        else:
            reward -= 0.1 * (word_count - 25)  # penalty for excessive length

        # 3. Task-specific semantic rewards (Regex-based target checking)
        if "başkent" in prompt_lower:
            # check if Capital of Turkey is answered correctly
            if "ankara" in response_lower:
                reward += 1.5
            else:
                reward -= 1.0
        
        if "yapay zeka" in prompt_lower:
            if any(kw in response_lower for kw in ["taklit", "bilgisayar", "sistem", "makine", "insan"]):
                reward += 1.0
                
        if "gezegen" in prompt_lower:
            if "jüpiter" in response_lower:
                reward += 1.5

        # 4. Readability and repetition checks
        words = response_lower.split()
        repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                repeats += 1
        if repeats > 0:
            reward -= 0.4 * repeats

        # 5. Büyük Ünlü Uyumu (Turkish Vowel Harmony Check)
        words_raw = re.findall(r'\b\w+\b', response_lower)
        harmony_count = 0
        total_valid_words = 0
        for word in words_raw:
            has_back = any(v in word for v in "aıou")
            has_front = any(v in word for v in "eiöü")
            if has_back or has_front:
                total_valid_words += 1
                if not (has_back and has_front):
                    harmony_count += 1
                    
        if total_valid_words > 0:
            harmony_ratio = harmony_count / total_valid_words
            # Reward high vowel harmony adherence
            reward += 0.5 * (harmony_ratio - 0.5)

        # 6. Türkçe Durum/Ek Uyum Denetimi (Case Suffix Harmony)
        suffix_matches = re.finditer(r'\b(\w+)\'([a-zA-ZçğıöşüÇĞİÖŞÜ]+)\b', response)
        for match in suffix_matches:
            stem = match.group(1).lower()
            suffix = match.group(2).lower()
            
            stem_vowels = [c for c in stem if c in "aeıioöuü"]
            if stem_vowels:
                last_v = stem_vowels[-1]
                is_last_v_back = last_v in "aıou"
                
                # Check case suffixes: de/da/den/dan/te/ta/ten/tan
                if suffix in ["da", "de", "dan", "den", "ta", "te", "dan", "den", "tan", "ten"]:
                    has_a = "a" in suffix
                    has_e = "e" in suffix
                    if is_last_v_back and has_e:
                        reward -= 0.5  # Mismatch (e.g. Ankara'de)
                    elif not is_last_v_back and has_a:
                        reward -= 0.5  # Mismatch (e.g. İzmir'da)
                    else:
                        reward += 0.3  # Correct match!

        return reward


def run_grpo(args):
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

    # Initialize model config and model (the Policy to optimize)
    config = SudyConfig(**config_dict)
    print("Initializing active Policy model...")
    policy_model = SudyLMHeadModel(config)

    # Load SFT base weights
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        print(f"Loading base SFT weights from {args.sft_checkpoint}...")
        policy_model.load_state_dict(torch.load(args.sft_checkpoint, map_location="cpu"))
    else:
        print("Warning: Starting GRPO without base SFT checkpoint!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)

    # Initialize reference model (with SFT weights - frozen)
    print("Initializing frozen Reference model...")
    ref_model = SudyLMHeadModel(config)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Prompts for RLHF
    prompts = [
        "türkiye'nin başkenti neresidir?",
        "yapay zeka nedir?",
        "güneş sistemindeki en büyük gezegen hangisidir?",
        "en hızlı koşan hayvan hangisidir?",
        "suyun formülü nedir?"
    ] * 20

    reward_manager = TurkishRewardManager()
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Starting GRPO training. Group Size G={args.group_size}, KL coef={args.kl_coef}")

    policy_model.train()
    
    # Loop over prompts
    # To implement GRPO:
    # 1. Prompt batch size
    # 2. For each prompt, generate G responses from active policy
    # 3. Calculate rewards
    # 4. Compute group-relative advantages
    # 5. Compute policy loss with KL constraint
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        total_reward = 0.0
        count = 0
        
        # Simple prompt loader
        for idx in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[idx : idx + args.batch_size]
            
            # Keep tracks of batch training tensors
            all_input_ids = []
            all_labels = []
            all_advantages = []
            all_ref_logprobs = []
            
            for prompt in batch_prompts:
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                # Keep prompt length
                prompt_len = len(prompt_ids)
                
                # Convert to tensor
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                
                # Generate G samples using policy
                group_responses_ids = []
                group_rewards = []
                
                # Set policy to eval for sampling
                policy_model.eval()
                with torch.no_grad():
                    for _ in range(args.group_size):
                        # Generate response IDs
                        generated = policy_model.generate(
                            prompt_tensor,
                            max_new_tokens=30,
                            temperature=0.7,
                            top_k=40
                        )
                        # generated is [1, prompt_len + generated_len]
                        full_ids = generated[0].tolist()
                        resp_ids = full_ids[prompt_len:]
                        
                        # Decode response text for reward scoring
                        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                        reward = reward_manager.evaluate(prompt, resp_text)
                        
                        group_responses_ids.append(full_ids)
                        group_rewards.append(reward)
                
                policy_model.train()
                
                # Normalize advantages for the group
                rewards_tensor = torch.tensor(group_rewards, dtype=torch.float)
                mean_r = rewards_tensor.mean()
                std_r = rewards_tensor.std() if len(rewards_tensor) > 1 else torch.tensor(1.0)
                if std_r < 1e-5:
                    std_r = torch.tensor(1.0)
                
                group_advantages = ((rewards_tensor - mean_r) / std_r).tolist()
                
                total_reward += mean_r.item()
                count += 1
                
                # Collect all samples in the group for batch backprop
                for i in range(args.group_size):
                    ids = group_responses_ids[i]
                    adv = group_advantages[i]
                    
                    # Pad to a fixed max length or handle dynamically.
                    # We will collect them and run a single forward pass over this group
                    all_input_ids.append(ids)
                    all_advantages.append(adv)
                    
                    # Label masks out prompt tokens (loss only calculated on output)
                    labels = [-100] * prompt_len + ids[prompt_len:]
                    all_labels.append(labels)
            
            if not all_input_ids:
                continue

            # Convert to padded tensors
            max_len = max(len(x) for x in all_input_ids)
            padded_input_ids = []
            padded_labels = []
            
            for ids, labels in zip(all_input_ids, all_labels):
                pad_len = max_len - len(ids)
                padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
                padded_labels.append(labels + [-100] * pad_len)
                
            input_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
            labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)
            advantages_tensor = torch.tensor(all_advantages, dtype=torch.float, device=device)
            
            # Compute reference model log-probabilities
            with torch.no_grad():
                ref_outputs = ref_model(input_tensor)
                ref_logits = ref_outputs["logits"]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                # Extract log-probs of actual tokens
                # [batch, seq_len - 1]
                ref_log_probs_selected = torch.gather(
                    ref_log_probs[:, :-1, :], 
                    dim=2, 
                    index=input_tensor[:, 1:].unsqueeze(-1)
                ).squeeze(-1)

            # Compute active policy model log-probabilities
            outputs = policy_model(input_tensor)
            logits = outputs["logits"]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_selected = torch.gather(
                log_probs[:, :-1, :], 
                dim=2, 
                index=input_tensor[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask out non-response tokens and padding tokens (-100 in labels)
            # labels_tensor starts from index 1 for comparison with shifted log probs
            loss_mask = (labels_tensor[:, 1:] != -100).float()
            
            # GRPO Objective:
            # ratio = exp(log_pi - log_pi_ref)
            ratio = torch.exp(log_probs_selected - ref_log_probs_selected)
            
            # Clip policy ratio
            surr1 = ratio * advantages_tensor.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages_tensor.unsqueeze(-1)
            
            # KL divergence penalty
            # KL approximated token-wise: exp(log_ref - log_pi) + log_pi - log_ref - 1
            # Or simpler: log_pi - log_ref
            kl = log_probs_selected - ref_log_probs_selected
            
            # Total Token loss
            policy_loss = -torch.min(surr1, surr2) + args.kl_coef * kl
            # Apply mask and average over response tokens
            masked_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            
            # Add aux model loss (routed expert load-balancing loss)
            total_loss = masked_loss + outputs["aux_loss"]

            # Step
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / max(1, count)
        avg_reward = total_reward / max(1, count)
        print(f"GRPO Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    config.save_pretrained(args.output_dir)
    print(f"Saved final GRPO-aligned model to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudy GRPO RLHF Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--sft_checkpoint", type=str, default="", help="Path to SFT model.pt")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sudy-rlhf", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of RLHF epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Prompt batch size")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size G")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")

    args = parser.parse_args()
    run_grpo(args)
