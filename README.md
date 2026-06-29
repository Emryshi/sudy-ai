# Sudy - Gelişmiş ve Verimli Türkçe LLM Kontrol Merkezi

<p align="center">
  <b>Gelişmiş MoE ve MLA mimari yeniliklerini Türkçe doğal dil işleme optimizasyonlarıyla buluşturan yüksek performanslı dil modeli projesi.</b>
</p>

---

## 📋 Proje Hakkında
Sudy, **hesaplama verimliliği, ölçeklenebilirlik ve Türkçe dil optimizasyonuna** odaklanarak sıfırdan tasarlanan açık kaynaklı bir Türkçe Büyük Dil Modeli (LLM) geliştirme projesidir. GPT-4 seviyesindeki performansı daha düşük maliyet ve yüksek bellek tasarrufuyla elde etmek için yenilikçi model mimarilerini (MLA, MoE, MTP) temel alır.

### 🚀 Temel Özellikler
*   **Çok Başlı Gizli Dikkat (MLA)**: KV (Key-Value) önbelleğini gizli bir uzaya sıkıştırarak çıkarım sırasında bellek kullanımını standart dikkat mekanizmalarına göre **~%90 azaltır**.
*   **Uzman Karışımı (MoE)**: Aktif uzmanlar ile her zaman çalışan paylaşılan uzmanları bir araya getiren seyrek mimari (Sparse MoE) sayesinde yüksek kapasiteyi düşük hesaplama maliyetiyle sunar.
*   **Çoklu-Token Tahmini (MTP)**: Sıralı veya paralel tahmin başlıkları yardımıyla modelin gelecekteki token temsillerini öğrenme yeteneğini geliştirir.
*   **GRPO (Group Relative Policy Optimization)**: Harici bir eleştirmen (critic) modeline gerek duymayan, grup içi çıktıların bağıl puanlamasıyla çalışan verimli pekiştirmeli öğrenme (RLHF) algoritması.
*   **Türkçe NLP Optimizasyonu**: Türkçe morfolojisine, ek yapılarına ve özel karakterlerine duyarlı BPE tokenizer ve veri ön işleme araçları.

---

## 📂 Depo Yapısı
```text
Sudy/
├── src/
│   ├── model/           # MLA, MoE, MTP mimari implementasyonları, tokenizer
│   ├── training/        # Pretrain, SFT, GRPO RLHF antrenman kodları
│   └── inference/       # FastAPI API sunucusu, batch çıkarım araçları
├── configs/             # YAML tabanlı model ve eğitim yapılandırmaları
├── scripts/             # Veri indirme, ön işleme, tokenizer eğitme betikleri
├── tests/               # pytest birim testleri
├── docs/                # Detaylı mimari, eğitim ve dağıtım kılavuzları
├── docker/              # Dockerfile ve docker-compose yapılandırması
└── README.md
```

---

## 🛠️ Kurulum

Yerel ortamda Sudy projesini kurmak için aşağıdaki adımları takip edebilirsiniz:

```bash
# Depoyu klonlayın ve içine girin
cd postercp

# Bağımlılıkları kurun
pip install -e .
```

---

## 📊 Çalıştırma ve Eğitim

Hızlıca örnek bir eğitim ve API başlatma süreci çalıştırmak için:

```bash
# 1. Örnek Türkçe veri kümesi oluşturun ve temizleyin
python scripts/download_data.py --source sample
python scripts/preprocess_data.py --input ./data/raw/turkish_sample.txt --output ./data/processed/turkish_processed.txt --deduplicate

# 2. Tokenizer'ı eğitin
python scripts/train_tokenizer.py --data_file ./data/processed/turkish_processed.txt --output_dir ./checkpoints/sudy-tokenizer --vocab_size 1000

# 3. Ön eğitimi (Pretraining) başlatın
python src/training/pretrain.py --config ./configs/pretrain.yaml --tokenizer_path ./checkpoints/sudy-tokenizer --data_path ./data/processed/turkish_processed.txt --output_dir ./checkpoints/sudy-pretrain --epochs 2 --batch_size 2

# 4. Denetimli İnce Ayarı (SFT) başlatın
python src/training/sft.py --config ./configs/sft.yaml --tokenizer_path ./checkpoints/sudy-tokenizer --pretrain_checkpoint ./checkpoints/sudy-pretrain/model.pt --output_dir ./checkpoints/sudy-sft --epochs 2 --batch_size 2

# 5. GRPO RLHF hizalamasını başlatın
python src/training/rlhf.py --config ./configs/rlhf.yaml --tokenizer_path ./checkpoints/sudy-tokenizer --sft_checkpoint ./checkpoints/sudy-sft/model.pt --output_dir ./checkpoints/sudy-rlhf --epochs 1 --batch_size 2
```

---

## 🌐 API Sunucusu

Eğitilen modelinizi FastAPI ile yayına almak için:

```bash
# Yönetim panelini ve API sunucusunu başlatın
python manage.py ui
```

---

## 🧪 Testlerin Çalıştırılması

Tüm birim testleri (unit tests) çalıştırmak için:
```bash
python manage.py test
```
