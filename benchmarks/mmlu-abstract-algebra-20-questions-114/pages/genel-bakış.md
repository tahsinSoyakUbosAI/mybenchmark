# Türkçe Çoktan Seçmeli Kıyaslama

Bu kıyaslama, yapay zeka modellerinin **Türkçe dil anlama** ve **çoktan seçmeli soru yanıtlama** yeteneklerini değerlendirmek için tasarlanmıştır.

---

## Kıyaslama Hakkında

**Amaç:** Modeliniz, çeşitli kategorilerde Türkçe çoktan seçmeli sorulara cevap verecektir. Her soru için A, B, C veya D şeklinde tek bir doğru cevap beklenmektedir.

**Örnek Soru:**

```
Kategori: Dilbilgisi

Soru: "Kitabı okudum" cümlesinde özne hangisidir?

   A) Kitabı
   B) Okudum
   C) Ben (gizli özne)
   D) Hiçbiri

Doğru Cevap: C
```

---

## Değerlendirme Metrikleri

### Ana Metrikler

| Metrik | Açıklama | Formül |
|--------|----------|--------|
| Genel Skor | Doğruluk ve LLM Judge ortalaması | (Doğruluk × 0.5) + (LLM Skoru × 0.5) |
| Doğruluk | Doğru cevaplanan soruların oranı | (Doğru / Toplam) × 100 |
| Kategori Skoru | Her kategori için ayrı başarı oranı | Dinamik hesaplama |

### Puanlama Sistemi

| Durum | Puan |
|-------|------|
| Doğru cevap | +1 |
| Yanlış cevap | 0 |
| Boş bırakma | 0 |

---

### LLM-as-Judge Metrikleri

> **Not:** Bu metrikler yalnızca `reasoning` alanı sağlandığında hesaplanır.

| Metrik | Açıklama | Aralık | İdeal |
|--------|----------|--------|-------|
| Anlamsal Tutarlılık | Gerekçenin mantıksal tutarlılığı | 0-100 | Yüksek |
| Akıcılık | Dilin doğallığı ve okunabilirliği | 0-100 | Yüksek |
| Cevap Gerekçesi | Cevabın ne kadar iyi desteklendiği | 0-100 | Yüksek |
| Düşünce Zinciri | Adım adım mantık tutarlılığı | 0-100 | Yüksek |
| Olgusal Tutarlılık | Bilgilerin doğruluğu | 0-100 | Yüksek |
| Halüsinasyon | Uydurma bilgi oranı | 0-100 | Düşük |
| LLM Skoru | Ağırlıklı ortalama | 0-100 | Yüksek |

---

## Nasıl Katılırsınız?

### Adım 1: Başlangıç Kiti
Sol menüden **"Dosyalar"** sekmesine gidin ve **Başlangıç Kiti**'ni indirin.

### Adım 2: Model Yapılandırması
`model.py` dosyasındaki LLM API ayarlarını yapılandırın:

```python
self.llm_config = {
    "api_url": "YOUR_API_URL/v1/chat/completions",
    "headers": {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    "model_name": "your-model-name",  # örn: gpt-4, gemini-pro
    "enabled": True
}
```

**Desteklenen API Formatları:**
- OpenAI uyumlu API'ler
- LiteLLM proxy
- Özel LLM sunucuları (OpenAI formatında)
- Azure OpenAI
- Google Gemini (OpenAI uyumlu)

### Adım 3: Gönderim Paketi
Dosyalarınızı ZIP formatında hazırlayın:

```
gonderim.zip
├── model.py          # Ana model dosyası (ZORUNLU)
├── metadata          # Çalıştırma komutu
└── (ek dosyalar)     # Gerekirse
```

### Adım 4: Gönderim
**"Katıl"** sekmesinden ZIP dosyanızı yükleyin.

---

## Gönderim Formatı

```json
{
    "predictions": [
        {
            "id": 1,
            "predicted_answer": "A",
            "category": "dilbilgisi",
            "confidence": 0.95,
            "reasoning": "Cümlede gizli özne 'ben' olduğu için C şıkkı doğrudur."
        },
        {
            "id": 2,
            "predicted_answer": "B",
            "category": "kelime_bilgisi",
            "confidence": 0.87,
            "reasoning": "'Sıla' kelimesi hasret anlamına gelir."
        }
    ]
}
```

### Alan Açıklamaları

| Alan | Zorunlu | Tip | Açıklama |
|------|---------|-----|----------|
| id | Evet | Integer | Soru numarası |
| predicted_answer | Evet | String | A, B, C veya D |
| category | Evet | String | Soru kategorisi |
| confidence | Hayır | Float | Güven skoru (0.0 - 1.0) |
| reasoning | Hayır | String | Cevap gerekçesi (LLM-Judge için) |

---

## Önemli Kurallar

**Yapılması Gerekenler:**
- Cevaplar BÜYÜK HARF olmalı (A, B, C, D)
- UTF-8 encoding kullanın
- Tüm soruları cevaplayın
- JSON formatına uyun

**Yapılmaması Gerekenler:**
- Küçük harf cevap vermeyin
- 60 saniyeyi aşmayın
- Geçersiz şık seçmeyin
- ID'leri karıştırmayın

---

*Başarılar dileriz! Sorularınız için iletişim sekmesini kullanabilirsiniz.*