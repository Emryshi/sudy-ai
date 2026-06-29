# 📋 Sudy - Verimli Türkçe LLM Proje Kılavuzu

Bu belge, gelişmiş mimari yenilikleri (MLA, MoE, MTP) temel alarak geliştirilen, düşük donanımlı sistemlerde bile çalışabilen ve siber güvenlik duvarıyla korunan **Sudy Türkçe Büyük Dil Modeli (LLM)** projesinin kurulum, eğitim, kullanım ve entegrasyon detaylarını içerir.

---

## 🏗️ 1. Sudy Mimarisi Nedir?

Sudy modeli sıfırdan verimlilik odaklı tasarlanmış bir modeldir. Temel mimari taşları şunlardır:
1.  **Çok Başlı Gizli Dikkat (Multi-Head Latent Attention - MLA)**: Anahtarları ve değerleri (KV) küçük bir gizli uzaya sıkıştırarak KV-Cache boyutunu **%75'e varan oranda azaltır**, böylece çıkarım hızı katlanır.
2.  **Uzmanların Karışımı (Mixture of Experts - MoE)**: Her token için sadece en alakalı uzmanlar (top-k) ve ortak uzmanlar (shared experts) çalıştırılır. Türkçe morfoloji tabanlı router (`MorphologicalMoERouter`), kelimenin ünlü uyumu ve isim/fiil ek yapısına göre yönlendirme yapar.
3.  **Çoklu-Token Tahmini (Multi-Token Prediction - MTP)**: Model sadece bir sonraki token'ı değil, $t+1$ ve $t+2$ token'larını da tahmin eder. Bu özellik çıkarımda **Spekülatif Çıkarım (Speculative Decoding)** ile tek adımda çift token üretmemizi sağlar.

---

## ⚙️ 2. Kurulum ve Gereksinimler

Projenin çalışması için Python 3.8+ gereklidir. Gerekli kütüphaneleri yüklemek için terminalde aşağıdaki komutu koşturun:
 
```bash
pip install -r requirements.txt
```

---

## 🚀 3. Modeli Nasıl Eğitiriz? (Training)

Sudy projesinde ön eğitimden pekiştirmeli öğrenmeye kadar uçtan uca bir eğitim hattı mevcuttur:

### A. Tokenizer (Alfabe) Eğitimi
Modelin kelimeleri doğru Türkçe hecelere bölebilmesi için Byte-Level BPE tokenizer'ı kendi veri setinizle eğitin:
```python
from src.model import SudyTokenizer

tokenizer = SudyTokenizer()
# Kendi Türkçe korpusunuzla eğitin
tokenizer.train(files=["data/raw/turkce_metinler.txt"], vocab_size=65536)
tokenizer.save("./checkpoints/sudy-tokenizer")
```

### B. Ön Eğitim (Pre-training)
Genel Türkçe dil yeteneklerini kazandırmak için ham metinler üzerinde causal model eğitimidir:
```bash
python src/training/pretrain.py --config configs/pretrain.yaml --tokenizer_path ./checkpoints/sudy-tokenizer --data_path data/raw/turkce_metinler.txt --output_dir ./checkpoints/sudy-pretrain --epochs 3 --batch_size 4 --lr 3e-4 --max_length 128
```

### C. Talimat İnce Ayarı (SFT - Supervised Fine-Tuning)
Modelin bir asistan gibi soru-cevap yapabilmesi için talimat veri setiyle eğitilmesidir:
```bash
python src/training/sft.py --config configs/sft.yaml --tokenizer_path ./checkpoints/sudy-tokenizer --pretrain_checkpoint ./checkpoints/sudy-pretrain/model.pt --data_path data/processed/sft_data.json --output_dir ./checkpoints/sudy-sft --epochs 3 --batch_size 4 --lr 5e-5
```

### D. Hizalama ve Pekiştirmeli Öğrenme (RLHF - GRPO)
Modelin düzgün Türkçe üretmesi, dilbilgisi kurallarına uyması ve yasadışı çıktılar üretmemesi amacıyla **GRPO (Group Relative Policy Optimization)** ile hizalanmasıdır. Ödül yöneticisinde Türkçe Büyük Ünlü Uyumu ve durum eki uyumu denetimleri mevcuttur:
```bash
python src/training/rlhf.py --config configs/rlhf.yaml --tokenizer_path ./checkpoints/sudy-tokenizer --sft_checkpoint ./checkpoints/sudy-sft/model.pt --output_dir ./checkpoints/sudy-rlhf --epochs 1 --batch_size 2 --group_size 4
```

---

## 💻 4. Web Kontrol Merkezi Kullanımı

Sohbet etmek, web sitelerinden otomatik eğitim yapmak veya yerel dosyaları yükleyip veya HuggingFace üzerinden veri indirip modeli eğitmek için tek bir arayüz mevcuttur.

Arayüzü başlatmak için:
```bash
python manage.py ui
```
Tarayıcınızdan `http://localhost:8000` adresine gidin. Arayüzde 4 ana bölüm yer alır:
1.  **💬 Sohbet Arayüzü**: Spekülatif Çıkarım, Web Arama (RAG) ve 8-bit Kuantizasyon seçenekleri kutucuklarla aktif edilerek modelle gerçek zamanlı (canlı akan yazıyla) sohbet edilir.
2.  **📁 Yerel Veri Yükleme ve HuggingFace İndirici**: Bilgisayarınızdaki `.csv`, `.json` veya `.txt` dosyalarını yükleyebilir veya doğrudan HuggingFace Hub repo id'si girerek veri kümelerini indirebilirsiniz.
3.  **🔗 Otomatik Eğitim (Linkten SFT)**: İstediğiniz bir web sayfasının adresini veya arama sorgusunu yapıştırıp eğitimi başlatabilirsiniz.
4.  **🧪 Birim Testleri**: Model mimarisinin ve çıkarım uçlarının bütünlüğünü doğrulamak için pytest birim testlerini koşturabilirsiniz.

---

## ⚡ 5. Düşük Kaynak (RAM/VRAM) Optimizasyonları

Sudy, ev bilgisayarlarında dahi çalışabilmesi için özel optimizasyonlar içerir:
*   **Ağırlık Kuantizasyonu (Int8 Kuantizasyon)**: Model ağırlıklarını 8-bit tamsayı (int8) hassasiyetinde sıkıştırarak bellek tüketimini **%50** azaltır.
*   **KV-Cache Kuantizasyonu**: MLA önbellek verileri bellek üzerinde 8-bit sıkıştırılmış olarak saklanır.
*   **LoRA (Low-Rank Adaptation) PEFT Eğitimi**: Eğitim sırasında modelin tüm ağırlıklarını dondurup sadece Attention katmanlarına küçük Rank-8 adaptör matrisleri enjekte eder.
