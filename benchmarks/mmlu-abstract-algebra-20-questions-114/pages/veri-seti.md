# Veri Formatı

Kıyaslama veri seti dokümantasyonu.

---

## Giriş Verisi (Input)

Veri seti **JSON** formatında sunulmaktadır.

### Soru Yapısı

```json
{
    "id": 1,
    "question": "Aşağıdaki cümledeki özne hangisidir?
'Küçük çocuk parkta oynuyor.'",
    "choices": {
        "label": ["A", "B", "C", "D"],
        "text": [
            "Küçük çocuk",
            "Parkta",
            "Oynuyor",
            "Küçük"
        ]
    },
    "category": "dilbilgisi"
}
```

### Alan Açıklamaları

| Alan | Tip | Açıklama | Örnek |
|------|-----|----------|-------|
| id | Integer | Benzersiz soru numarası | 1, 2, 3 |
| question | String | Soru metni | "Özne hangisidir?" |
| choices.label | Array | Şık etiketleri | ["A", "B", "C", "D"] |
| choices.text | Array | Şık içerikleri | ["Seçenek 1", ...] |
| category | String | Soru kategorisi | "dilbilgisi" |

---

## Çıkış Verisi (Output)

### Beklenen Format

```json
{
    "predictions": [
        {
            "id": 1,
            "predicted_answer": "A",
            "category": "dilbilgisi",
            "confidence": 0.92,
            "reasoning": "Cümlede 'Küçük çocuk' ifadesi öznedir çünkü eylemi yapan varlıktır."
        },
        {
            "id": 2,
            "predicted_answer": "C",
            "category": "kelime_bilgisi",
            "confidence": 0.78,
            "reasoning": "'Hasret' kelimesinin eş anlamlısı 'özlem'dir."
        }
    ]
}
```

### Zorunlu Alanlar

| Alan | Açıklama |
|------|----------|
| id | Sorunun ID'si (test verisindeki ile birebir eşleşmeli) |
| predicted_answer | Tahmin edilen cevap (A, B, C veya D) |
| category | Soru kategorisi (test verisindeki ile aynı) |

### Opsiyonel Alanlar

| Alan | Açıklama |
|------|----------|
| confidence | Modelin güven skoru (0.0 - 1.0 arası ondalık sayı) |
| reasoning | LLM-as-Judge değerlendirmesi için cevap gerekçesi |

---

## Kategoriler

> **Not:** Kategoriler Excel dosyanızdaki `category` sütununa göre otomatik oluşturulur.

### Örnek Kategoriler

| Kategori | Açıklama | Örnek Soru |
|----------|----------|------------|
| dilbilgisi | Türkçe dil bilgisi kuralları | Özne, yüklem, nesne belirleme |
| kelime_bilgisi | Kelime anlamları ve eş/zıt anlamlılar | Eş anlamlı kelime bulma |
| okuma_anlama | Metin anlama ve yorumlama | Paragraf soruları |
| yazim_kurallari | İmla ve noktalama | Doğru yazım seçimi |
| anlam_bilgisi | Cümle ve kelime anlamı | Anlam ilişkileri |

---

## Kontrol Listesi

- Tüm sorular cevaplanmalıdır
- Cevaplar BÜYÜK HARF olmalıdır (A, B, C, D)
- ID'ler test verisi ile birebir eşleşmelidir
- UTF-8 encoding kullanılmalıdır
- JSON formatı geçerli olmalıdır
- reasoning alanı LLM-Judge için önerilir

---

*Daha fazla bilgi için Genel Bakış sayfasına göz atın.*