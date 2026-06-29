# Türkçe Doğal Dil İşleme (NLP) Optimizasyonları

Türkçe, morfolojik olarak zengin ve eklemeli (agglutinative) bir dildir. Bu nedenle, genel amaçlı dil modeli mimarilerinde Türkçe verimliliğini artırmak için özel tasarımlar yapılması gerekir. Sudy projesinde uyguladığımız bazı Türkçe NLP optimizasyonları şunlardır:

## 1. Morfolojik Farkındalıklı BPE Tokenizer
Türkçe'de kelimelere eklenen yapım ve çekim ekleri kelime uzunluğunu ve çeşitliliğini aşırı derecede artırır. Standart İngilizce odaklı tokenizer modelleri Türkçe kelimeleri anlamsız alt parçalara (subwords) böler.
- **Apostrof Ayrımı (Apostrophe Splitter)**: Özel isimlere gelen çekim eklerini kesme işareti (`'`) hizasından bölerek kök kelime ile eklerin bağımsız olarak öğrenilmesini sağlıyoruz (Örn: `İstanbul'da` -> `["İstanbul", "'", "da"]`).
- **Genişletilmiş Kelime Dağarcığı (Vocab Size: 65,536)**: Türkçe morfolojik varyasyonları kapsamak adına kelime dağarcığını 65,536 olarak yapılandırıyoruz. Bu sayede yaygın kök ve ek kombinasyonları tek bir token halinde temsil edilebilir.

## 2. Türkçe Karakter Duyarlı Temizleyici (Text Cleaner)
Python'ın varsayılan `.lower()` fonksiyonu Türkçe `I` ve `İ` harflerini İngilizce kurallarına göre (`i` ve `i`) dönüştürür. Bu durum kelime anlamlarının kaybolmasına sebep olur (Örn: `IRK` -> `ırk` yerine `irk`).
- Sudy veri temizleme hattında `TextCleaner.lowercase_tr` fonksiyonunu kullanarak Türkçe büyük-küçük harf dönüşüm kurallarını tam olarak uyguluyoruz.
- Unicode NFKC normalizasyonu ile Türkçe diakritik işaretlerini (ç, ğ, ı, ö, ş, ü) ve şapkalı sesli harfleri (â, î, û) standardize ediyoruz.

## 3. Kural Tabanlı Ödül Modeli Hizalaması (GRPO Alignment)
Hizalama (Alignment) aşamasında, modelin Türkçe gramer yapısına uygun yanıtlar üretmesi ve tekrara düşmesini engellemek için Türkçe kurallarla zenginleştirilmiş ödül fonksiyonları kullanıyoruz.
- Cümle başlarında büyük harfle başlama ve cümle sonlarında uygun noktalama işareti kullanma kuralları puanlanır.
- Türkçe ek ardışıklıklarında oluşabilecek tekrarlama çöküşlerini engellemek için kelime seviyesinde ceza puanı (penalty) uygulanır.
