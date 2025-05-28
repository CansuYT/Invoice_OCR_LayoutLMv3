## Fatura Verisi Çıkarımı (OCR ve LayoutLMv3 Fine-Tuning)

Bu proje, **fatura PDF'lerinden** yapılandırılmış veri çıkarımı yapmayı amaçlar. Proje, **OCR** (Optical Character Recognition) ve önceden eğitimli **LayoutLM** modeli ile fine-tune tekniklerini kullanarak, fatura verilerinden bilgilerin çıkarılmasını sağlar.

## 📌 Proje Özeti

Bu projede, **fatura PDF'lerinden** verileri çıkarmak için pytesseract kütüphanesi ile **OCR** ve transformers kütüphanesi ile **LayoutLM modeli** kullanılır.

Proje, fatura verilerinin **dinamik düzenleri** ile çalışacak şekilde tasarlanmıştır. Ayrıca, çıkarılan verilerin doğruluğu ve tutarlılığı üzerinde yoğunlaşılır.

Projenin adımları şöyledir:

- 1. Faturalar resme dönüştürülüp OCR işlemi gerçekleştirilerek resimlerden metin çıkarılır.
- 2. Elde edilen metinler, model eğitimi için uygun formata getirilir.İstenilen çıktılar doğrultusunda, etiketler ve kelime konum bilgileri dahil edilerek veriseti oluşturulur.
- 3. Özel etiketler ve hazırlanan veriseti ile LayoutLM modeli eğitilir (fine-tune edilir).
- 4. Eğitilen model ile faturadan istenilen bilgiler çıkartılarak rapor edilir.

Sonuç olarak bu projede amaç, performansı yüksek LayoutLMv3 modelini, kendi verisetimize uyaralayarak kişiselleştirilmiş bir fatura bilgi çıkarım modeli oluşturmaktır.

*Proje, beklenen performansı genellikle sağlıyor ancak bazı durumlarda daha iyi sonuçlar elde edilebilir. Geliştirmeler yapılarak daha yüksek doğruluklar elde edilebilir.*



