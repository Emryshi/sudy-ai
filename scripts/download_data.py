import os
import argparse
import requests
from tqdm import tqdm

def download_file(url: str, dest: str):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading {url} to {dest}...")
    with open(dest, 'wb') as file, tqdm(
        desc=os.path.basename(dest),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    parser = argparse.ArgumentParser(description="Sudy Data Downloader")
    parser.add_argument("--source", type=str, default="sample", choices=["sample", "wikipedia", "mc4"], help="Data source to download")
    parser.add_argument("--output_dir", type=str, default="./data/raw", help="Target raw directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "sample":
        print("Generating a local sample Turkish corpus file...")
        sample_path = os.path.join(args.output_dir, "turkish_sample.txt")
        samples = [
            "türkçe dil yapısı morfolojik olarak eklemeli diller grubuna girer.",
            "yapay zeka ve makine öğrenimi modern dünyada büyük dönüşümler yapmaktadır.",
            "büyük dil modelleri derin öğrenme mimarilerinin gelişmesiyle ortaya çıkmıştır.",
            "doğal dil işleme alanında türkçe kaynakların artması kritik öneme sahiptir.",
            "sudy model ailesi yüksek verimlilik odaklı mimari yapısıyla bilinir.",
            "veri setlerindeki gürültülerin temizlenmesi modellerin kalitesini doğrudan etkiler.",
            "türkiye'nin bilişim ekosistemi yapay zeka alanında hızla büyümektedir.",
            "açık kaynaklı projeler yazılım dünyasının gelişimini hızlandırmaktadır."
        ] * 100
        with open(sample_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(s + "\n")
        print(f"Sample corpus created at {sample_path}")
    else:
        # Placeholder for real datasets URLs
        print(f"To download real {args.source} datasets, please run with python scripts/download_data.py or integrate HuggingFace datasets library directly.")

if __name__ == "__main__":
    main()
