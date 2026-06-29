import os
import argparse
import sys
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='ignore')

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.search_utils import WebSearcher, is_safe_url
from src.training.data_utils import TextCleaner, MinHashLSH

def main():
    parser = argparse.ArgumentParser(description="Sudy Autonomous Web Crawler Agent")
    parser.add_argument("--query", type=str, required=True, help="Web search query to find sources for")
    parser.add_argument("--max_results", type=int, default=3, help="Max search results to crawl")
    parser.add_argument("--output", type=str, default="data/processed/autonomous_facts.txt", help="Output text file to save facts")
    args = parser.parse_args()

    print(f"Autonomous Web Crawler started for query: '{args.query}'")
    
    searcher = WebSearcher()
    results = searcher.search(args.query, max_results=args.max_results)
    
    if not results:
        print("No search results found or query blocked.")
        return

    print(f"Found {len(results)} potential source pages. Starting autonomous crawl...")

    lsh = MinHashLSH(threshold=0.7)
    crawled_count = 0
    facts_count = 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Append mode so we collect data incrementally
    with open(args.output, "a", encoding="utf-8") as outfile:
        for item in results:
            url = item["url"]
            title = item["title"]
            
            if not is_safe_url(url):
                print(f"Skipping dangerous or local URL: {url}")
                continue
                
            print(f"Starting deep crawl for: {title} ({url})")
            
            # Recursive domain-restricted crawl loop
            visited = set()
            to_visit = [(url, 0)]
            parsed_start = urlparse(url)
            start_domain = parsed_start.netloc
            
            while to_visit:
                curr_url, depth = to_visit.pop(0)
                
                if curr_url in visited:
                    continue
                visited.add(curr_url)
                
                if not is_safe_url(curr_url):
                    continue
                    
                print(f"Crawling (Depth {depth}/2): {curr_url}")
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(curr_url, headers=headers, timeout=8)
                    if response.status_code != 200:
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Strip style/script/nav
                    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        s.decompose()
                        
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) < 30:
                            continue
                            
                        cleaned = TextCleaner.clean(text)
                        if cleaned and not lsh.is_duplicate(cleaned):
                            outfile.write(cleaned + "\n")
                            facts_count += 1
                            
                    # Add child links on same domain
                    if depth < 2:
                        for link in soup.find_all('a', href=True):
                            href = link['href'].strip()
                            full_url = urljoin(curr_url, href)
                            full_url = full_url.split('#')[0]
                            
                            parsed_full = urlparse(full_url)
                            if parsed_full.netloc == start_domain and full_url not in visited:
                                to_visit.append((full_url, depth + 1))
                                
                except Exception as e:
                    print(f"Error crawling {curr_url} at depth {depth}: {e}")
                    
            crawled_count += 1

    print("\nAutonomous Web Crawling completed successfully!")
    print(f"Successfully crawled pages: {crawled_count}/{len(results)}")
    print(f"New unique facts added to {args.output}: {facts_count}")

if __name__ == "__main__":
    main()
