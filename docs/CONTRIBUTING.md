# Katkıda Bulunma Rehberi

Sudy projesine katkıda bulunmak istediğiniz için teşekkür ederiz! Açık kaynak topluluğunun gücüyle Türkçe dil modellerini geliştirmek en temel vizyonumuzdur.

## Nasıl Katkı Sağlarsınız?

### 1. Hata Bildirimi (Bug Report)
Bir hata tespit ettiğinizde lütfen GitHub Issues bölümünden hata bildirim şablonunu doldurarak yeni bir bildirim açın. Hatanın nasıl tekrarlanacağını (reproduce) net adımlarla açıklayın.

### 2. Özellik Önerileri (Feature Request)
Projeye yeni bir mimari katman, veri temizleme aracı veya optimizasyon eklemek istiyorsanız bir Issue açarak fikrinizi toplulukla paylaşın ve tartışın.

### 3. Kod Katkısı (Pull Request)
Kod katkısı sağlamak için:
1. Depoyu forklayın (Fork).
2. Yeni bir özellik dalı (branch) açın: `git checkout -b feature/yeniozellik`.
3. Değişikliklerinizi yapın ve pytest testlerini çalıştırarak kodun doğruluğundan emin olun (`pytest tests/`).
4. Kod formatını `black` ile biçimlendirin (`black src/ tests/`).
5. Dalınızı commitleyin ve pushlayın.
6. Bir Pull Request (PR) açın.

## Kod Standartları
- Tüm Python kodları [Black](https://github.com/psf/black) formatına uygun olmalıdır.
- Kodunuzu test eden uygun pytest birim testlerini `tests/` dizini altına ekleyin.
- Kod içi yorumlarda ve dokümantasyonda anlaşılır bir Türkçe/İngilizce teknik dil kullanın.
