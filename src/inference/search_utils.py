import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict
import math
import socket
from urllib.parse import urlparse, parse_qs, unquote
import ipaddress

def is_safe_url(url: str) -> bool:
    """
    Checks if a URL is safe from SSRF (Server-Side Request Forgery).
    Filters out private, local, and metadata IP addresses.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # Enforce default web ports to prevent internal port scanning
        port = parsed.port
        if port and port not in [80, 443]:
            return False
            
        # Resolve hostname to IP
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
        
        # Block private, link-local, loopback IPs (e.g. localhost, AWS metadata server)
        if (ip.is_loopback or 
            ip.is_private or 
            ip.is_link_local or 
            ip.is_multicast or 
            ip.is_unspecified):
            print(f"Security Alert: Blocked dangerous URL target mapping to: {ip}")
            return False
            
        return True
    except Exception:
        return False

class WebSearcher:
    """
    Simulates real-time web browsing.
    Queries search engines and retrieves raw page contents.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def search(self, query: str, max_results: int = 2) -> List[Dict[str, str]]:
        """Queries DuckDuckGo HTML interface and returns list of organic results (title, link, snippet)."""
        print(f"Searching web for: '{query}'")
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        
        try:
            response = requests.get(search_url, headers=self.headers, timeout=8)
            if response.status_code != 200:
                print("Search query failed.")
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # DuckDuckGo HTML page selector for organic results
            for a in soup.find_all('a', class_='result__snippet', limit=max_results):
                result_div = a.find_parent('div', class_='result__body')
                if not result_div:
                    continue
                    
                title_link = result_div.find('a', class_='result__url')
                if not title_link:
                    continue
                    
                title = title_link.get_text().strip()
                url = title_link.get('href', '').strip()
                snippet = a.get_text().strip()
                
                # Resolve DuckDuckGo redirect URLs if any
                if "uddg=" in url:
                    parsed_redirect = urlparse(url)
                    query_params = parse_qs(parsed_redirect.query)
                    if "uddg" in query_params:
                        url = query_params["uddg"][0]
                
                if url.startswith("//"):
                    url = "https:" + url
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
                
            return results
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def fetch_page_text(self, url: str) -> str:
        """Download raw text paragraphs from a target webpage with SSRF checks."""
        if not url or not is_safe_url(url):
            return ""
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code != 200:
                return ""
                
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
                
            paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
            # Join paragraphs that contain actual sentences
            clean_text = "\n".join([p for p in paragraphs if len(p.split()) > 5])
            return clean_text
        except Exception as e:
            print(f"Failed to fetch page text for {url}: {e}")
            return ""


class ContextCompressor:
    """
    Filters and compresses large search results to select only the most relevant paragraphs.
    Saves token budget and VRAM during prompt construction.
    """
    @staticmethod
    def get_lexical_score(query: str, paragraph: str) -> float:
        """Computes basic word-overlap/tf-idf-like relevance score between query and paragraph."""
        query_words = set(re.findall(r'\w+', query.lower()))
        paragraph_words = re.findall(r'\w+', paragraph.lower())
        if not paragraph_words:
            return 0.0
            
        # Count matches
        match_count = sum(1 for w in paragraph_words if w in query_words)
        
        # Length penalty to prevent long irrelevant blocks from scoring high
        length_penalty = 1.0 / (1.0 + 0.001 * len(paragraph_words))
        
        # Word density / overlap score
        score = (match_count / len(query_words)) * length_penalty
        return score

    def compress(self, query: str, documents: List[str], max_tokens_approx: int = 300) -> str:
        """Splits documents into paragraphs, ranks them, and keeps only the top relevant blocks."""
        all_paragraphs = []
        for doc in documents:
            paragraphs = [p.strip() for p in doc.split('\n') if p.strip()]
            all_paragraphs.extend(paragraphs)
            
        # Score and rank paragraphs
        scored_paragraphs = []
        for p in all_paragraphs:
            score = self.get_lexical_score(query, p)
            if score > 0.0:  # must have at least one word overlap
                scored_paragraphs.append((score, p))
                
        # Sort by score descending
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        
        # Accumulate top paragraphs under budget
        selected_paragraphs = []
        token_count = 0
        
        for score, p in scored_paragraphs:
            approx_tokens = len(p.split())
            if token_count + approx_tokens > max_tokens_approx:
                if len(selected_paragraphs) == 0:
                    # Keep at least one paragraph even if over budget
                    selected_paragraphs.append(p)
                break
            selected_paragraphs.append(p)
            token_count += approx_tokens
            
        if not selected_paragraphs:
            return "İnternet aramasında doğrudan eşleşen detaylı bilgi bulunamadı."
            
        return "\n\n".join(selected_paragraphs)
