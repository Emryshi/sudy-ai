# Sudy - Gelişmiş ve Verimli Türkçe LLM Kontrol Merkezi

<p align="center">
  <b>MLA · MoE · MTP — Türkçe doğal dil işleme için optimize edilmiş, düşük VRAM tüketimiyle yüksek performanslı açık kaynak dil modeli.</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.95%2B-green?logo=fastapi">
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue">
</p>

---

## Hızlı Başlangıç

```bash
# 1. Depoyu klonlayın
git clone https://github.com/Emryshi/sudy-ai.git
cd sudy-ai

# 2. Bağımlılıkları kurun
pip install -e .

# 3. Tokenizer eğitin ve web panelini başlatın
python manage.py setup
python manage.py ui
```

Tarayıcınızda **http://localhost:8000** adresine giderek kontrol paneline erişebilirsiniz.

---

## Proje Hakkında

Sudy, **hesaplama verimliliği, ölçeklenebilirlik ve Türkçe dil optimizasyonuna** odaklanarak sıfırdan tasarlanan açık kaynaklı bir Türkçe Büyük Dil Modeli (LLM) geliştirme çerçevesidir.

### Temel Özellikler

| Özellik | Açıklama |
|---|---|
| **MLA (Çok Başlı Gizli Dikkat)** | KV önbelleğini gizli uzaya sıkıştırarak çıkarım belleğini **~%90 azaltır** |
| **MoE (Uzman Karışımı)** | Türkçe morfolojisine duyarlı yönlendirici ile her token için top-k uzman seçer |
| **MTP (Çoklu-Token Tahmini)** | Bir forward pass'te birden fazla token üreterek çıkarım hızını **2x artırır** |
| **GRPO RLHF** | Harici critic modeli gerektirmeden grup-içi bağıl politika optimizasyonu |
| **Web Arama (RAG)** | DuckDuckGo üzerinden gerçek zamanlı bilgi çekimi ve bağlam sıkıştırma |
| **Otonom Crawler** | Belirtilen URL'leri derinlemesine gezerek kendi kendine veri toplar |

---

## Depo Yapısı

```text
sudy-ai/
├── src/
│   ├── model/           # MLA, MoE, MTP mimari implementasyonları ve tokenizer
│   ├── training/        # Pretrain, SFT, GRPO RLHF eğitim betikleri
│   └── inference/       # FastAPI API sunucusu, güvenlik katmanı, web arama
├── configs/             # YAML tabanlı model konfigürasyonları
│   ├── pretrain.yaml    # Test / hızlı eğitim (küçük boyut)
│   └── model_full.yaml  # Üretim modeli (1B+ parametre)
├── scripts/             # Veri indirme, crawling, HuggingFace downloader
├── tests/               # 23 adet pytest birim testi
├── docs/                # Detaylı kılavuzlar
└── docker/              # CUDA destekli Dockerfile ve GPU compose
```

---

## Kurulum ve Eğitim

```bash
# Tokenizer eğitimi
python manage.py setup --vocab_size 8000

# Ön eğitim (Pretraining)
python manage.py pretrain --epochs 3 --batch_size 4

# Denetimli ince ayar (SFT)
python manage.py sft --epochs 2

# GRPO RLHF hizalaması
python manage.py rlhf --epochs 1
```

### HuggingFace'den Veri Kümesi İndirme

Web panelindeki **"Veri Kümeleri"** sekmesinden HuggingFace Hub'daki herhangi bir veri kümesini doğrudan indirebilirsiniz. Örnek:

```
Repo ID: boun-tabi-LMG/turkish-instructions
Split:   train
```

---

## Web Paneli

```bash
python manage.py ui   # http://localhost:8000
```

- **Sohbet:** Gerçek zamanlı streaming yanıtlar, web arama (RAG) modu
- **Veri Kümeleri:** CSV/JSON yükleme, HuggingFace indirici, içerik önizleme
- **Model Eğitimi:** Tüm eğitim aşamalarını canlı log konsoluyla başlatma
- **Birim Testleri:** pytest'i tek tıkla çalıştırma

---

## Docker ile Çalıştırma

```bash
# CPU modu
docker-compose -f docker/docker-compose.yml up --build

# GPU modu (NVIDIA Container Toolkit gerekli)
docker-compose -f docker/docker-compose.yml up --build
# docker-compose.yml içindeki deploy.resources bloğu GPU'yu otomatik aktifleştirir
```

---

## Testlerin Çalıştırılması

```bash
python manage.py test
# veya doğrudan:
pytest tests/ -v
```

23 birim testi: model mimarisi, tokenizer, MLA dikkat mekanizması, MoE yük dengeleme, RLHF, API uç noktaları.
