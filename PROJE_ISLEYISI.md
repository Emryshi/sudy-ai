# Sudy LLM - Proje İşleyiş Kılavuzu

Bu kılavuz, Sudy Türkçe Büyük Dil Modeli (LLM) projesinin teknik mimarisini, eğitim aşamalarını, web paneli üzerindeki işleyişini ve otonom veri toplama mekanizmalarını açıklamaktadır.

---

## 🏗️ 1. Model Mimarisi

Sudy, Türkçe dil yapısına optimize edilmiş ve düşük donanımlı cihazlarda bile yüksek performansla çalışacak şekilde tasarlanmış özgün bir dil modelidir.

### A. Çok Başlı Gizli Dikkat (Multi-Head Latent Attention - MLA)
Standart Transformer modellerindeki dikkat (Attention) mekanizmalarında, her yeni token üretildiğinde önceki anahtar ve değer (Key-Value) vektörleri hafızada saklanır (KV-Cache). Bu durum uzun bağlamlarda yüksek VRAM tüketimine sebep olur.
*   **Çözüm:** MLA, KV vektörlerini düşük boyutlu gizli bir uzaya (Latent Space) sıkıştırarak çıkarım sırasındaki bellek kullanımını **%90 oranında düşürür**.

### B. Uzmanların Karması (Mixture of Experts - MoE)
Her token için tüm yapay sinir ağı katmanlarının çalıştırılması yerine, sadece belirli uzman katmanlar (Experts) tetiklenir.
*   **Morphological MoE Router:** Türkçe morfolojik yapısına (Büyük Ünlü Uyumu, ek çekimleri vb.) duyarlı bir yönlendirici içerir. Gelen kelimenin yapısına göre en uygun uzmanları (`top-k` uzman + `shared` sabit uzman) seçerek hesaplama maliyetini düşürürken bilgi kapasitesini maksimumda tutar.

### C. Çoklu Token Tahmini (Multi-Token Prediction - MTP)
Model sadece sıradaki tokenı ($t+1$) değil, paralel veya sıralı tahmin başlıklarıyla sonraki tokenları da ($t+2$, $t+3$) eşzamanlı olarak tahmin etmeye çalışır.
*   **Spekülatif Çıkarım:** Çıkarım (inference) aşamasında tek bir ileri geçişte (forward pass) birden fazla token üretilmesini sağlayarak üretim hızını 2 kata kadar artırır.

---

## 📊 2. Eğitim İşleyiş Hattı (Training Pipeline)

Sudy'de 4 aşamalı bir eğitim ve hizalama akışı mevcuttur:

1.  **Kurulum ve Tokenizer (Setup):** Byte-Level BPE tokenizer, Türkçe ek ve hece yapısını öğrenebilmesi için Türkçe korpus ile eğitilir.
2.  **Ön Eğitim (Pre-training):** Ham Türkçe metinler üzerinde nedensel dil modellemesi (Causal LM) yapılarak temel dil yetenekleri kazandırılır.
3.  **Talimat İnce Ayarı (SFT):** Soru-cevap (Instruction-Response) verileriyle model bir yapay zeka asistanı formuna sokulur.
4.  **GRPO Pekiştirmeli Öğrenme (RLHF Alignment):** Modelin çıktılarının Türkçe dil kurallarına (ünlü uyumu, doğru ek kullanımı) ve siber güvenlik sınırlarına uyması için grup bazlı bağıl politika optimizasyonu (GRPO) uygulanır.

---

## 🌐 3. Web Yönetim Paneli Özellikleri

Proje, `python manage.py ui` komutuyla başlatılan şık ve minimalist bir web paneli üzerinden yönetilir:

### A. Yapay Zeka Sohbet Ekranı
*   **Canlı Akış (Streaming Response):** Yanıtlar kullanıcıya gerçek zamanlı olarak kelime kelime akar.
*   **Web Arama (RAG):** DuckDuckGo üzerinden sorguyu aratarak gelen sonuçların en alakalı kısımlarını bağlam olarak kullanır. Yönlendirme (redirect) URL'lerini ayrıştırarak kararlı çalışır.
*   **Düşük RAM Modu:** 8-bit kuantizasyon desteğiyle modeli sıkıştırarak çalıştırır.

### B. Veri Kümeleri Yönetimi
*   **Yerel Dosya Yükleme:** Sürükle-bırak yöntemiyle CSV veya JSON dosyaları yüklenebilir. Yüklenen veriler otomatik olarak `data/processed/` klasörü altına kaydedilir.
*   **HuggingFace İndirici:** HuggingFace Hub üzerindeki herhangi bir veri kümesi (repo_id, split ve limit belirtilerek) doğrudan panele indirilip SFT JSON veya Pretrain TXT formatına dönüştürülebilir.

### C. Gelişmiş Otonom İnternet Taraması (Self-Training)
*   **Recursive Crawler:** Girilen web linklerinin veya arama sorgusu sonuçlarının sadece ilk sayfalarını değil, aynı alan adına bağlı iç sayfalarını da derinlemesine gezerek benzersiz verileri toplar, tekilleştirir (LSH ile) ve modeli eğitmek üzere depolar.
*   **JSON & CSV Desteği:** Eğitim betikleri hem JSON hem de CSV uzantılı veri kümelerini doğrudan okuyabilir, Türkçe ve İngilizce sütun adlarını otomatik eşleştirir.
