# Sudy Eğitim Kılavuzu

Sudy Türkçe LLM modeli, dört farklı aşamadan oluşan bir eğitim hattı kullanır. Bu kılavuzda, yerel geliştirme veya çoklu GPU ortamında bu adımların nasıl çalıştırılacağı anlatılmaktadır.

## 1. Hazırlık ve Tokenizer Eğitimi
Verilerinizi `./data/raw/` dizinine yerleştirin.
```bash
# Örnek Türkçe veri kümesi oluşturun
python scripts/download_data.py --source sample

# Veriyi temizleyin ve tekilleştirin
python scripts/preprocess_data.py \
    --input ./data/raw/turkish_sample.txt \
    --output ./data/processed/turkish_processed.txt \
    --deduplicate

# Türkçe morfolojik BPE Tokenizer'ı eğitin
python scripts/train_tokenizer.py \
    --data_file ./data/processed/turkish_processed.txt \
    --output_dir ./checkpoints/sudy-tokenizer \
    --vocab_size 1000
```

## 2. Ön Eğitim (Pretraining)
Modelin dil bilgisi ve genel kültürü öğrenmesini sağlayan ilk aşamadır.
```bash
# Ön eğitimi başlatın
python src/training/pretrain.py \
    --config ./configs/pretrain.yaml \
    --tokenizer_path ./checkpoints/sudy-tokenizer \
    --data_path ./data/processed/turkish_processed.txt \
    --output_dir ./checkpoints/sudy-pretrain \
    --epochs 3 \
    --batch_size 2
```

## 3. Denetimli İnce Ayar (Supervised Fine-Tuning - SFT)
Modelin talimatları takip edebilmesi ve soru-cevap yapabilmesi için denetimli veri setleri ile eğitildiği aşamadır.
```bash
# SFT aşamasını başlatın
python src/training/sft.py \
    --config ./configs/sft.yaml \
    --tokenizer_path ./checkpoints/sudy-tokenizer \
    --pretrain_checkpoint ./checkpoints/sudy-pretrain/model.pt \
    --output_dir ./checkpoints/sudy-sft \
    --epochs 3 \
    --batch_size 2
```

## 4. Pekiştirmeli Öğrenme (GRPO RLHF)
GRPO (Group Relative Policy Optimization) algoritması ile insan tercihlerine ve kurallara uygun şekilde model hizalamasının yapıldığı aşamadır.
```bash
# GRPO hizalamasını başlatın
python src/training/rlhf.py \
    --config ./configs/rlhf.yaml \
    --tokenizer_path ./checkpoints/sudy-tokenizer \
    --sft_checkpoint ./checkpoints/sudy-sft/model.pt \
    --output_dir ./checkpoints/sudy-rlhf \
    --epochs 1 \
    --batch_size 2 \
    --group_size 4
```
> [!NOTE]
> GRPO aşaması, her bir istem (prompt) için 4 adet aday yanıt üretir ve bunları yerleşik Türkçe Ödül Yöneticisi (`TurkishRewardManager`) ile puanlayarak modeli günceller.
