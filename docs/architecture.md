# Sudy Mimari Tasarımı

Bu belgede Sudy dil modelinin arkasındaki mimari yenilikler ve çalışma prensipleri açıklanmaktadır.

## 1. Çok Başlı Gizli Dikkat (Multi-Head Latent Attention - MLA)
Geleneksel Çok Başlı Dikkat (MHA) mekanizmalarında, KV (Key-Value) önbelleği çıkarım aşamasında ciddi bir bellek darboğazı yaratır. MLA, Key ve Value vektörlerini düşük boyutlu bir gizli uzaya sıkıştırarak bu darboğazı çözmeyi hedefler.

- **KV Sıkıştırma (KV Compression)**: 
  İlgili adımda gizli durum $h_t$ sıkıştırılarak $c^{KV}_t$ oluşturulur:
  $$c^{KV}_t = \text{RMSNorm}(W_{down\_kv} h_t)$$
  Sıkıştırma oranı genellikle 8x civarındadır ($d_{latent} = 512$ ve $d_{model} = 4096$).
- **Geri Açma (Up-projection)**:
  Matematiksel işlemlerde $c^{KV}_t$ matrisi tekrar $k^C_t$ ve $v^C_t$ olarak açılır.
- **Konum Kodlama (Decoupled RoPE)**:
  Sıkıştırılmış anahtar-değer matrislerine doğrudan Döner Konum Gömme (RoPE) uygulanamayacağı için, konum bilgisi taşıyan ek bir anahtar $k^R_t$ ve sorgu $q^R_t$ katmanı ayrılarak dikkat hesaplamasına eklenir.

Bu sayede standart dikkat hesaplamasına göre KV önbellek boyutu **~%90** oranında azaltılır.

## 2. Uzman Karışımı (Mixture of Experts - MoE)
Sudy, seyrek (sparse) FFN mimarisi kullanarak hesaplama maliyetini düşürürken model kapasitesini artırır.

- **Yönlendirme (Routing)**:
  Her token için bir yönlendirici (router) ağ geçidi, uzmanların uygunluk skorlarını hesaplar ve en yüksek skora sahip ilk $K$ uzmanı ($K=2$ veya $K=1$) seçer.
- **Paylaşılan Uzmanlar (Shared Experts)**:
  Seçilen dinamik uzmanların yanı sıra, her zaman aktif olan paylaşılan uzmanlar (shared experts) bulunur. Bu uzmanlar genel bilgiyi toplayarak modelin stabilitesini artırır.
- **Yük Dengeleme Kaybı (Load-Balancing Loss)**:
  Uzmanların çömesini (tüm tokenların aynı uzmana gitmesini) engellemek için eğitim aşamasında yardımcı bir kayıp fonksiyonu (auxiliary loss) hesaplanır.

## 3. Çoklu-Token Tahmini (Multi-Token Prediction - MTP)
Sudy, sadece $t+1$. tokenı değil, sonraki tokenları da ($t+2$, $t+3$) tahmin edecek paralel/sıralı tahmin başlıklarına sahiptir.

- Eğitim sırasında $h_t$ vektörü ile bir sonraki tokenın gömme vektörü $E_{t+1}$ birleştirilerek MTP başlığına beslenir.
- MTP başlıkları, modelin gelecekteki tokenlar hakkında daha zengin temsiller öğrenmesini sağlar.
