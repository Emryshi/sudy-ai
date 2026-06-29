import os
import csv
import json
import random
import argparse

# General conversational dataset template library
GENERAL_CONVERSATIONS = [
    # Greetings
    ("merhaba", "merhaba! ben sudy, size nasıl yardımcı olabilirim?"),
    ("selam", "selam! size yardımcı olmaktan mutluluk duyarım. ne hakkında konuşmak istersiniz?"),
    ("nasılsın", "çok teşekkür ederim, harikayım! size hizmet etmek için hazırım. siz nasılsınız?"),
    ("naber", "iyiyim, teşekkürler. umarım sizin için de her şey yolundadır. nasıl yardımcı olabilirim?"),
    ("günaydın", "günaydın! harika bir gün geçirmenizi dilerim. bugün sizin için ne yapabilirim?"),
    ("iyi akşamlar", "iyi akşamlar! umarım gününüz güzel geçmiştir. nasıl yardımcı olabilirim?"),
    ("iyi geceler", "iyi geceler! dinlendirici bir uyku dilerim. sormak istediğiniz bir şey var mıydı?"),
    
    # Identity & Creator
    ("adın ne", "benim adım sudy. türkçe dil yapısına özel olarak geliştirilmiş bir yapay zeka asistanıyım."),
    ("sen kimsin", "ben sudy, türkçe konuşabilen, türkçe dil yapısı ve gramer kurallarına hakim gelişmiş bir büyük dil modeliyim."),
    ("hangi dili konuşuyorsun", "ben türkçe konuşuyorum. türkçe dil kuralları, morfolojisi ve zengin kelime haznesi üzerine eğitildim."),
    ("kim seni eğitti", "ben deepmind/sudy ai ekibi tarafından geliştirilen ve eğitilen yerli bir yapay zeka dil modeliyim."),
    ("yapay zeka mısın", "evet, ben yapay zeka tabanlı bir büyük dil modeliyim. insan dilini anlamak ve akıcı cevaplar vermek üzere tasarlandım."),
    
    # Gratitude
    ("teşekkürler", "rica ederim! yardımcı olabildiysem ne mutlu bana. başka bir sorunuz var mı?"),
    ("teşekkür ederim", "rica ederim, her zaman yardımcı olmaktan mutluluk duyarım!"),
    ("sağ ol", "siz de sağ olun! her zaman buradayım."),
    ("harikasın", "çok teşekkür ederim! siz de harikasınız. size yardımcı olmak benim için bir zevk."),
    
    # Basic Knowledge & RAG prompts
    ("türkiye'nin başkenti neresidir", "türkiye'nin başkenti ankara'dır."),
    ("yapay zeka nedir", "yapay zeka (ai), insan zekasını taklit ederek veri analizi yapabilen, öğrenebilen, çıkarım yapabilen ve karmaşık problemleri çözebilen bilgisayar tabanlı akıllı sistemlerdir."),
    ("en hızlı koşan hayvan hangisidir", "dünyanın en hızlı koşan kara hayvanı çitadır. avlanırken saatte 100-120 km hıza kadar ulaşabilir."),
    ("suyun formülü nedir", "suyun kimyasal formülü h2o'dur; iki hidrojen ve bir oksijen atomundan oluşur."),
    ("güneş sistemindeki en büyük gezegen hangisidir", "güneş sistemimizdeki en büyük gezegen jüpiter'dir."),
    ("büyük ünlü uyumu nedir", "büyük ünlü uyumu, türkçe kökenli kelimelerde kalın ünlü harflerden (a, ı, o, u) sonra kalın ünlülerin, ince ünlü harflerden (e, i, ö, ü) sonra da ince ünlülerin gelmesi kuralıdır. örnek: 'araba' (kalın uyumlu), 'sevgi' (ince uyumlu)."),
    ("küçük ünlü uyumu nedir", "küçük ünlü uyumu, türkçe kelimelerdeki ünlülerin düzlük-yuvarlaklık ve darlık-genişlik özelliklerine göre birbirini takip etmesi kuralıdır. buna göre düz ünlülerden (a, e, ı, i) sonra düz ünlüler; yuvarlak ünlülerden (o, ö, u, ü) sonra ya düz-geniş (a, e) ya da dar-yuvarlak (u, ü) ünlüler gelebilir."),
    
    # Logic / Coding
    ("python nedir", "python, okunabilirliği yüksek, genel amaçlı, nesne yönelimli ve dinamik tipli açık kaynaklı bir programlama dilidir. veri bilimi, yapay zeka ve web geliştirme alanlarında yaygın olarak tercih edilir."),
    ("makine öğrenimi nedir", "makine öğrenimi, bilgisayarların açıkça programlanmadan verilerden öğrenmesini ve deneyim yoluyla kararlar almasını sağlayan bir yapay zeka alt dalıdır.")
]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('I', 'ı').replace('İ', 'i')
    return text.strip().lower()

def main():
    parser = argparse.ArgumentParser(description="Sudy Advanced Dataset Generator")
    parser.add_argument("--csv_path", type=str, default="data/raw/tdk.csv", help="Path to tdk.csv")
    parser.add_argument("--output_json", type=str, default="data/processed/sudy_sft_advanced.json", help="Output SFT dataset file")
    parser.add_argument("--output_txt", type=str, default="data/processed/sudy_pretrain_advanced.txt", help="Output Pretrain dataset file")
    parser.add_argument("--max_samples", type=int, default=100000, help="Maximum number of SFT rows to generate")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: Raw TDK CSV file not found at {args.csv_path}")
        print("Please upload or ensure tdk.csv is in data/raw/")
        return

    print(f"Reading dictionary data from {args.csv_path}...")
    
    raw_entries = []
    # Read the CSV file cleanly, resolving encoding variations
    for encoding in ['utf-8', 'iso-8859-9', 'windows-1254']:
        try:
            with open(args.csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    madde = row.get("madde", "").strip()
                    anlam = row.get("anlam", "").strip()
                    ornek = row.get("ornek", "").strip()
                    if madde and anlam:
                        raw_entries.append({
                            "madde": madde,
                            "anlam": anlam,
                            "ornek": ornek
                        })
            print(f"Loaded {len(raw_entries)} entries using encoding: {encoding}")
            break
        except Exception as e:
            raw_entries = []
            continue

    if not raw_entries:
        print("Error: Could not load any valid entries from CSV.")
        return

    # Shuffle to distribute different categories evenly
    random.seed(42)
    random.shuffle(raw_entries)

    sft_dataset = []
    pretrain_passages = []

    # 1. Inject general conversations (weighted)
    print("Injecting general conversations...")
    for _ in range(50):  # Augment conversational weight
        for prompt, resp in GENERAL_CONVERSATIONS:
            sft_dataset.append({
                "prompt": prompt,
                "response": resp
            })
            pretrain_passages.append(f"{prompt} {resp}")

    # 2. Process entries into SFT and pretraining formats
    print("Generating Q&A prompts, example sentences, and RAG context dialogs...")
    
    # Split entries into subgroups to build diverse tasks
    num_entries = len(raw_entries)
    
    # We will generate up to max_samples SFT rows.
    for idx, entry in enumerate(raw_entries):
        if len(sft_dataset) >= args.max_samples:
            break

        madde = entry["madde"]
        anlam = entry["anlam"]
        ornek = entry["ornek"]

        # Pretraining text generation
        desc_text = f"'{madde}' kelimesinin türkçe sözlük anlamı '{anlam}' şeklindedir."
        if ornek:
            desc_text += f" örnek kullanım cümlesi: '{ornek}'"
        pretrain_passages.append(desc_text)

        # Distribute into different templates
        task_mod = idx % 10

        if task_mod < 4:
            # Task A: Meaning queries (40% weight)
            templates = [
                (f"\"{madde}\" ne demektir?", f"\"{madde}\" kelimesi, \"{anlam}\" anlamına gelmektedir."),
                (f"\"{madde}\" kelimesinin türkçe anlamı nedir?", f"\"{madde}\" kelimesinin anlamı: {anlam}."),
                (f"\"{madde}\" ne anlama gelir?", f"\"{madde}\" ifadesi şu anlama gelir: {anlam}."),
                (f"bana \"{madde}\" kelimesinin tanımını yapar mısın?", f"tabii ki. \"{madde}\" ifadesinin tanımı: {anlam}."),
                (f"\"{madde}\" sözlük anlamı nedir?", f"\"{madde}\" sözlük anlamı: {anlam}.")
            ]
            q, a = random.choice(templates)
            sft_dataset.append({"prompt": q, "response": a})

        elif task_mod < 7:
            # Task B: Example usage queries (30% weight)
            if ornek:
                templates = [
                    (f"\"{madde}\" kelimesini bir cümlede kullanabilir misin?", f"elbette, işte örnek bir cümle: \"{ornek}\""),
                    (f"\"{madde}\" ifadesiyle ilgili örnek bir cümle kurar mısın?", f"tabii, şu şekilde bir cümle kurabiliriz: \"{ornek}\""),
                    (f"bana \"{madde}\" içeren örnek bir cümle verir misin?", f"memnuniyetle. işte örnek cümle: \"{ornek}\"")
                ]
                q, a = random.choice(templates)
                sft_dataset.append({"prompt": q, "response": a})
            else:
                sft_dataset.append({
                    "prompt": f"\"{madde}\" kelimesi ne anlama gelmektedir?",
                    "response": f"\"{madde}\" kelimesinin anlamı: {anlam}."
                })

        else:
            # Task C: RAG / Web Search context queries (30% weight)
            if ornek:
                context = f"internet arama sonuçları:\n[1] bilgi: \"{madde}\" kelimesinin sözlükteki karşılığı \"{anlam}\" şeklindedir. kullanım örneği: \"{ornek}\""
                prompt = f"{context}\n\nsoru: arama sonuçlarına dayanarak, {madde} kelimesinin anlamı ve bir örnek cümlesi nedir?\ncevap:"
                response = f"arama sonuçlarına göre, \"{madde}\" kelimesi \"{anlam}\" anlamına gelmektedir. örnek kullanım cümlesi ise şu şekildedir: \"{ornek}\""
            else:
                context = f"internet arama sonuçları:\n[1] sözlük tanımına göre, \"{madde}\" ifadesi \"{anlam}\" anlamına gelir."
                prompt = f"{context}\n\nsoru: internet aramasına göre {madde} ne anlama gelir?\ncevap:"
                response = f"arama sonuçlarına göre, \"{madde}\" ifadesi \"{anlam}\" anlamına gelmektedir."
            
            sft_dataset.append({"prompt": prompt, "response": response})

    # Save SFT JSON file
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=4)

    # Save Pretrain TXT file
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        for passage in pretrain_passages:
            f.write(passage + "\n")

    print("\nDataset preparation completed successfully!")
    print(f"Saved SFT dataset to: {args.output_json} (Count: {len(sft_dataset)})")
    print(f"Saved Pretrain text corpus to: {args.output_txt} (Passages: {len(pretrain_passages)})")

if __name__ == "__main__":
    main()
