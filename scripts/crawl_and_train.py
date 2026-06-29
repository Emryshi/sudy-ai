import os
import argparse
import requests
import re
import json
import torch
import yaml
import sys
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='ignore')

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer
from src.training.data_utils import TextCleaner, MinHashLSH, SudyDataset
from src.training.sft import train_sft
from src.inference.search_utils import is_safe_url

def crawl_url_recursive(start_url: str, max_depth: int = 2) -> list:
    """Fetch and scrape clean text from a URL and recursively crawl internal links up to max_depth."""
    visited = set()
    to_visit = [(start_url, 0)]
    content_blocks = []
    
    parsed_start = urlparse(start_url)
    start_domain = parsed_start.netloc
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    while to_visit:
        url, depth = to_visit.pop(0)
        
        if url in visited:
            continue
        visited.add(url)
        
        if not is_safe_url(url):
            print(f"Security Warning: Skipping unsafe target URL: {url}")
            continue
            
        print(f"Crawling (Depth {depth}/{max_depth}): {url}")
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
                
            # Extract headings and paragraphs
            for child in soup.find_all(['h1', 'h2', 'h3', 'p']):
                text = child.get_text().strip()
                if len(text) > 15:
                    content_blocks.append({
                        "tag": child.name,
                        "text": text
                    })
                    
            # Queue child links on same domain
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href'].strip()
                    full_url = urljoin(url, href)
                    full_url = full_url.split('#')[0]
                    
                    parsed_full = urlparse(full_url)
                    if parsed_full.netloc == start_domain and full_url not in visited:
                        to_visit.append((full_url, depth + 1))
                        
        except Exception as e:
            print(f"Error crawling {url} at depth {depth}: {e}")
            
    return content_blocks

def generate_qa_pairs(content_blocks):
    """
    Automated self-instruction parser:
    Generates Q&A pairs using HTML structure (headings as prompts, paragraphs as answers).
    """
    qa_pairs = []
    current_prompt = None
    
    for block in content_blocks:
        tag = block["tag"]
        text = block["text"]
        
        if tag in ['h1', 'h2', 'h3']:
            # Normalise heading to look like a query
            query = TextCleaner.clean(text)
            if query:
                current_prompt = f"{query} hakkında bilgi verir misin?"
        elif tag == 'p' and len(text.split()) > 8:
            answer = TextCleaner.clean(text)
            if current_prompt and answer:
                qa_pairs.append({
                    "prompt": current_prompt,
                    "response": answer
                })
            elif answer:
                # If no heading was found, generate a generic prompt from sentence start
                # E.g. "yapay zeka...", Prompt: "yapay zeka nedir?"
                words = answer.split()
                short_subject = " ".join(words[:3])
                qa_pairs.append({
                    "prompt": f"{short_subject} nedir?",
                    "response": answer
                })
                
    return qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Sudy Web Scraper & Auto-Trainer")
    parser.add_argument("--urls", type=str, required=True, help="Comma-separated list of target URLs to crawl")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--sft_checkpoint", type=str, default="", help="Path to SFT model.pt (base checkpoint)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sudy-auto-train", help="Directory to save auto-trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    # 1. Crawl targeted URLs
    url_list = [u.strip() for u in args.urls.split(",") if u.strip()]
    all_qa_pairs = []
    lsh = MinHashLSH(threshold=0.7)
    
    print(f"Starting crawl phase for {len(url_list)} websites...")
    for url in url_list:
        blocks = crawl_url_recursive(url)
        if not blocks:
            continue
            
        qa_list = generate_qa_pairs(blocks)
        for qa in qa_list:
            # Deduplicate questions/answers using MinHash
            if not lsh.is_duplicate(qa["response"]):
                all_qa_pairs.append(qa)

    print(f"\nCrawling & processing completed. Extracted {len(all_qa_pairs)} unique instruction pairs.")

    if len(all_qa_pairs) == 0:
        print("Error: No training pairs could be extracted. Exiting.")
        import sys
        sys.exit(1)

    # Save to data directory
    os.makedirs("./data/processed", exist_ok=True)
    crawled_data_file = "./data/processed/crawled_sft.json"
    with open(crawled_data_file, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
    print(f"Extracted dataset saved to {crawled_data_file}")

    # 2. Trigger automated training step
    print("\nStarting SFT training step on crawled data...")
    
    # We mock args structure to pass to SFT train module
    class SFTArgs:
        def __init__(self):
            self.config = args.config
            self.tokenizer_path = args.tokenizer_path
            self.pretrain_checkpoint = args.sft_checkpoint
            self.data_path = crawled_data_file
            self.output_dir = args.output_dir
            self.epochs = args.epochs
            self.batch_size = 2
            self.lr = args.lr
            self.max_length = 256
            
    train_sft(SFTArgs())
    print(f"\nSelf-training completed successfully. Auto-trained model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
