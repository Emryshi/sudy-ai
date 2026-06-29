#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import shutil
import threading

# Try to use emojis if the console supports it, otherwise fallback to text tags
try:
    "🚀".encode(sys.stdout.encoding or 'utf-8')
    OK = "🟢"
    INFO = "🔵"
    WARN = "🟡"
    ERR = "🔴"
    ROCKET = "🚀"
    SUCCESS = "🎉"
except Exception:
    OK = "[OK]"
    INFO = "[INFO]"
    WARN = "[WARN]"
    ERR = "[ERR]"
    ROCKET = "[RUN]"
    SUCCESS = "[SUCCESS]"


def run_command(cmd, env_updates=None):
    """Utility to run a subprocess and stream output."""
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    
    print(f"\n{INFO} Çalıştırılan komut: {' '.join(cmd)}")
    try:
        # Run subprocess and stream output directly to terminal
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n{ERR} Komut başarısız oldu! Çıkış kodu: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n{ERR} Bir hata oluştu: {e}")
        return False

def cmd_setup(args):
    print(f"\n{ROCKET} 1. Adım: Örnek Türkçe veri kümesi oluşturuluyor...")
    if not run_command([sys.executable, "scripts/download_data.py", "--source", "sample"]):
        return False

    print(f"\n{ROCKET} 2. Adım: Veri kümesi temizleniyor ve tekilleştiriliyor (LSH)...")
    if not run_command([
        sys.executable, "scripts/preprocess_data.py",
        "--input", "./data/raw/turkish_sample.txt",
        "--output", "./data/processed/turkish_processed.txt",
        "--deduplicate"
    ]):
        return False

    print(f"\n{ROCKET} 3. Adım: Türkçe morfoloji uyumlu tokenizer eğitiliyor...")
    if not run_command([
        sys.executable, "scripts/train_tokenizer.py",
        "--data_file", "./data/processed/turkish_processed.txt",
        "--output_dir", "./checkpoints/sudy-tokenizer",
        "--vocab_size", str(args.vocab_size)
    ]):
        return False

    print(f"\n{SUCCESS} Kurulum başarıyla tamamlandı! Veriler ve tokenizer hazır.")
    return True

def cmd_pretrain(args):
    print(f"\n{ROCKET} Causal Ön Eğitim (Pretraining) başlatılıyor...")
    cmd = [
        sys.executable, "src/training/pretrain.py",
        "--config", args.config,
        "--tokenizer_path", args.tokenizer_path,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--max_length", str(args.max_length)
    ]
    if run_command(cmd):
        print(f"\n{SUCCESS} Ön eğitim başarıyla tamamlandı! Model kaydedildi: {args.output_dir}")
        return True
    return False

def cmd_sft(args):
    print(f"\n{ROCKET} Denetimli İnce Ayar (SFT) başlatılıyor...")
    # Check if pretrain checkpoint exists
    if not os.path.exists(args.pretrain_checkpoint):
        print(f"{WARN} Ön eğitim ağırlıkları bulunamadı: {args.pretrain_checkpoint}")
        print(f"{INFO} SFT sıfırdan başlatılacak veya test verileri kullanılacaktır.")
        
    cmd = [
        sys.executable, "src/training/sft.py",
        "--config", args.config,
        "--tokenizer_path", args.tokenizer_path,
        "--pretrain_checkpoint", args.pretrain_checkpoint,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--max_length", str(args.max_length)
    ]
    if run_command(cmd):
        print(f"\n{SUCCESS} SFT başarıyla tamamlandı! Model kaydedildi: {args.output_dir}")
        return True
    return False

def cmd_rlhf(args):
    print(f"\n{ROCKET} GRPO RLHF hizalaması başlatılıyor...")
    if not os.path.exists(args.sft_checkpoint):
        print(f"{WARN} SFT ağırlıkları bulunamadı: {args.sft_checkpoint}")
        
    cmd = [
        sys.executable, "src/training/rlhf.py",
        "--config", args.config,
        "--tokenizer_path", args.tokenizer_path,
        "--sft_checkpoint", args.sft_checkpoint,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--group_size", str(args.group_size),
        "--lr", str(args.lr),
        "--kl_coef", str(args.kl_coef),
        "--clip_eps", str(args.clip_eps)
    ]
    if run_command(cmd):
        print(f"\n{SUCCESS} GRPO RLHF eğitimi başarıyla tamamlandı! Model kaydedildi: {args.output_dir}")
        return True
    return False

def cmd_test(args):
    print(f"\n{ROCKET} pytest birim testleri çalıştırılıyor...")
    cmd = ["pytest"]
    if args.verbose:
        cmd.append("-v")
    if args.filter:
        cmd.extend(["-k", args.filter])
        
    return run_command(cmd)

def cmd_api(args):
    print(f"\n{ROCKET} FastAPI çıkarım (inference) sunucusu başlatılıyor...")
    env_updates = {
        "MODEL_PATH": args.model_path,
        "TOKENIZER_PATH": args.tokenizer_path,
        "HOST": args.host,
        "PORT": str(args.port)
    }
    # Run uvicorn server pointing to package style module
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.inference.api:app", 
        "--host", args.host, 
        "--port", str(args.port)
    ]
    if args.reload:
        cmd.append("--reload")
        
    return run_command(cmd, env_updates=env_updates)

def cmd_ui(args):
    print(f"\n{ROCKET} Sudy Özel Web Yönetim Paneli başlatılıyor...")
    
    # Start background thread to open browser automatically
    def open_browser():
        import time
        import webbrowser
        time.sleep(2.0)
        print(f"\n{INFO} Tarayıcı otomatik açılıyor: http://127.0.0.1:8000")
        webbrowser.open("http://127.0.0.1:8000")
        
    t = threading.Thread(target=open_browser)
    t.daemon = True
    t.start()
    
    class APIArgs:
        def __init__(self):
            self.model_path = args.model_path
            self.tokenizer_path = args.tokenizer_path
            self.host = "127.0.0.1"
            self.port = 8000
            self.reload = False
            
    return cmd_api(APIArgs())

def cmd_clean(args):
    print(f"\n{ROCKET} Proje geçici dosyaları temizleniyor...")
    paths_to_remove = [
        ".pytest_cache",
        "src/__pycache__",
        "src/model/__pycache__",
        "src/training/__pycache__",
        "src/inference/__pycache__",
        "tests/__pycache__",
        "build",
        "dist",
        "sudy.egg-info"
    ]
    for p in paths_to_remove:
        if os.path.exists(p):
            print(f"{INFO} Siliniyor: {p}")
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    print(f"\n{SUCCESS} Temizlik tamamlandı!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Sudy Türkçe LLM Yönetim ve Çalıştırma Aracı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Gerçekleştirilecek eylem")
    
    # 1. Setup
    p_setup = subparsers.add_parser("setup", help="Örnek verileri hazırlar ve tokenizer'ı eğitir.")
    p_setup.add_argument("--vocab_size", type=int, default=1000, help="Tokenizer kelime haznesi boyutu")
    p_setup.set_defaults(func=cmd_setup)
    
    # 2. Pretrain
    p_pretrain = subparsers.add_parser("pretrain", help="Model ön eğitimini (Pretraining) başlatır.")
    p_pretrain.add_argument("--config", default="configs/pretrain.yaml", help="Model mimari yapılandırma dosyası")
    p_pretrain.add_argument("--tokenizer_path", default="checkpoints/sudy-tokenizer", help="Tokenizer dizini")
    p_pretrain.add_argument("--data_path", default="data/processed/sudy_pretrain_advanced.txt", help="Temiz metin veri yolu")
    p_pretrain.add_argument("--output_dir", default="checkpoints/sudy-pretrain", help="Model çıktı dizini")
    p_pretrain.add_argument("--epochs", type=int, default=2, help="Epoch sayısı")
    p_pretrain.add_argument("--batch_size", type=int, default=2, help="Batch boyutu")
    p_pretrain.add_argument("--lr", type=float, default=3e-4, help="Öğrenme oranı (Learning rate)")
    p_pretrain.add_argument("--max_length", type=int, default=128, help="Maksimum dizi uzunluğu")
    p_pretrain.set_defaults(func=cmd_pretrain)
    
    # 3. SFT
    p_sft = subparsers.add_parser("sft", help="Denetimli İnce Ayarı (SFT) başlatır.")
    p_sft.add_argument("--config", default="configs/sft.yaml", help="Model yapılandırma dosyası")
    p_sft.add_argument("--tokenizer_path", default="checkpoints/sudy-tokenizer", help="Tokenizer dizini")
    p_sft.add_argument("--pretrain_checkpoint", default="checkpoints/sudy-pretrain/model.pt", help="Ön eğitimli model ağırlıkları")
    p_sft.add_argument("--data_path", default="data/processed/sudy_sft_advanced.json", help="SFT eğitim veri yolu (Boş bırakılırsa sentetik veri üretilir)")
    p_sft.add_argument("--output_dir", default="checkpoints/sudy-sft", help="Model çıktı dizini")
    p_sft.add_argument("--epochs", type=int, default=2, help="Epoch sayısı")
    p_sft.add_argument("--batch_size", type=int, default=2, help="Batch boyutu")
    p_sft.add_argument("--lr", type=float, default=5e-5, help="SFT öğrenme oranı")
    p_sft.add_argument("--max_length", type=int, default=256, help="Maksimum dizi uzunluğu")
    p_sft.set_defaults(func=cmd_sft)
    
    # 4. RLHF
    p_rlhf = subparsers.add_parser("rlhf", help="GRPO RLHF hizalamasını başlatır.")
    p_rlhf.add_argument("--config", default="configs/rlhf.yaml", help="Model yapılandırma dosyası")
    p_rlhf.add_argument("--tokenizer_path", default="checkpoints/sudy-tokenizer", help="Tokenizer dizini")
    p_rlhf.add_argument("--sft_checkpoint", default="checkpoints/sudy-sft/model.pt", help="SFT modeli ağırlıkları")
    p_rlhf.add_argument("--output_dir", default="checkpoints/sudy-rlhf", help="Model çıktı dizini")
    p_rlhf.add_argument("--epochs", type=int, default=1, help="RLHF epoch sayısı")
    p_rlhf.add_argument("--batch_size", type=int, default=2, help="Batch boyutu")
    p_rlhf.add_argument("--group_size", type=int, default=4, help="GRPO grup boyutu G")
    p_rlhf.add_argument("--lr", type=float, default=1e-6, help="RLHF öğrenme oranı")
    p_rlhf.add_argument("--kl_coef", type=float, default=0.01, help="KL ceza katsayısı")
    p_rlhf.add_argument("--clip_eps", type=float, default=0.2, help="PPO kırpma katsayısı")
    p_rlhf.set_defaults(func=cmd_rlhf)
    
    # 5. Test
    p_test = subparsers.add_parser("test", help="Birim testleri (Pytest) çalıştırır.")
    p_test.add_argument("-v", "--verbose", action="store_true", help="Detaylı test çıktısı")
    p_test.add_argument("-k", "--filter", help="Belirli bir testi filtrelemek için ifade")
    p_test.set_defaults(func=cmd_test)
    
    # 6. API
    p_api = subparsers.add_parser("api", help="FastAPI sunucusunu başlatır.")
    p_api.add_argument("--model_path", default="checkpoints/sudy-sft", help="Sunulacak modelin ağırlık dizini")
    p_api.add_argument("--tokenizer_path", default="checkpoints/sudy-tokenizer", help="Tokenizer dizini")
    p_api.add_argument("--host", default="0.0.0.0", help="Sunucu adresi")
    p_api.add_argument("--port", type=int, default=8000, help="Sunucu portu")
    p_api.add_argument("--reload", action="store_true", help="Uvicorn otomatik yeniden yüklemeyi etkinleştir")
    p_api.set_defaults(func=cmd_api)
    
    # 7. UI
    p_ui = subparsers.add_parser("ui", help="Sudy özel web yönetim panelini başlatır.")
    p_ui.add_argument("--model_path", default="checkpoints/sudy-sft", help="Yüklenecek modelin ağırlık dizini")
    p_ui.add_argument("--tokenizer_path", default="checkpoints/sudy-tokenizer", help="Tokenizer dizini")
    p_ui.set_defaults(func=cmd_ui)
    
    # 8. Clean
    p_clean = subparsers.add_parser("clean", help="Geçici ve cache dosyalarını temizler.")
    p_clean.set_defaults(func=cmd_clean)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    success = args.func(args)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
