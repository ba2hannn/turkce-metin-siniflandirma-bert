# Türkçe Metin Sınıflandırma Projesi

Bu proje, Türkçe metinleri ırkçılık, kızdırma, nötr ve cinsiyetçilik gibi kategorilere ayırmak için eğitilmiş bir BERT modelini kullanır. Model, sosyal medya yorumları, haberler ve diğer metin tabanlı içeriklerdeki toksik dil kullanımını tespit etmek amacıyla geliştirilmiştir.

## Modelin Performansı

Modelin başarısını göstermek amacıyla, aşağıda bazı performans metrikleri ve grafikler sunulmaktadır. Bu grafikler, modelin farklı kategorilerdeki başarısını ve genel performansını görselleştirmektedir.

### ROC Eğrileri (Çoklu Sınıflandırma)

![ROC Eğrileri](https://github.com/ba2hannn/turkce-metin-siniflandirma-bert/blob/main/Figure_3.png)

*   **Açıklama:** Çoklu sınıflı ROC eğrisi, her bir sınıf için modelin doğru pozitif oranını (True Positive Rate) yanlış pozitif oranına (False Positive Rate) karşı nasıl dengelediğini gösterir. Model, tüm sınıflarda yüksek performans göstermektedir.
*   **AUC Değerleri:** Her bir sınıf için elde edilen AUC (Area Under Curve) değerleri, modelin sınıflandırma başarısının ne kadar yüksek olduğunu gösterir.

### Karışıklık Matrisi (Confusion Matrix)

![Karışıklık Matrisi](https://github.com/ba2hannn/turkce-metin-siniflandirma-bert/blob/main/Figure_3.png)

*   **Açıklama:** Karışıklık matrisi, modelin tahmin ettiği sınıfları gerçek sınıflara karşı gösterir. Bu matris, modelin hangi sınıfları karıştırabildiğini ve hangi sınıflarda daha iyi performans gösterdiğini anlamamızı sağlar.

### Model Değerlendirme Metrikleri

![Model Değerlendirme Metrikleri](https://github.com/ba2hannn/turkce-metin-siniflandirma-bert/blob/main/Figure_4.png)

*   **Açıklama:** Bu grafik, modelin her sınıf için kesinlik (Precision), geri çağırma (Recall) ve F1-skoru (F1 Score) değerlerini gösterir. Bu metrikler, modelin başarısının genel bir özetini sunar ve modelin ne kadar dengeli performans gösterdiğini anlamamızı sağlar.
    *   **Precision (Kesinlik):** Modelin pozitif olarak tahmin ettiği örneklerin ne kadarının gerçekten pozitif olduğunu gösterir.
    *   **Recall (Geri Çağırma):** Gerçek pozitif örneklerin ne kadarının model tarafından doğru olarak tahmin edildiğini gösterir.
    *   **F1 Score:** Precision ve recall'un harmonik ortalamasıdır ve modelin performansının dengeli bir ölçüsünü sunar.

## Modelin Çalışma Mantığı

Bu model, Hugging Face'in sağladığı `bert-base-turkish-uncased` modelini temel alır ve belirli bir veri seti üzerinde duygusal sınıflandırma için ince ayarlanmıştır. Model, aşağıdaki adımları izleyerek çalışır:

1.  **Tokenizasyon:** Metin girdileri, modelin anlayabileceği sayısal tokenlara (kelime parçalarına) dönüştürülür.
2.  **Girdi Oluşturma:** Tokenlar, modele girdi olarak beslenmek üzere uygun bir formata getirilir.
3.  **Model İleri Besleme:** BERT modeli, tokenları işleyerek her sınıf için bir olasılık puanı hesaplar.
4.  **Sınıflandırma:** En yüksek olasılık puanına sahip sınıf, tahmin edilen sınıf olarak belirlenir.

## Kullanım

Aşağıda, bu modelin nasıl kullanılacağına dair basit bir örnek sunulmuştur:

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from huggingface_hub import hf_hub_download

# Model ve tokenizer'ı Hugging Face Hub'dan yükle
model_adi = "ba2hann/bert_base_turkish_sentiment_analysis"
tokenizer = BertTokenizer.from_pretrained(model_adi)
model = BertForSequenceClassification.from_pretrained(model_adi)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sonuçlar için kategori eşlemesi
kategori_haritasi = {
    "Irkçılık": 0,
    "Kızdırma": 1,
    "Nötr": 2,
    "Cinsiyetçilik": 3
}
ters_kategori_haritasi = {v: k for k, v in kategori_haritasi.items()}


def metni_analiz_et(metin):
    # Metni tokenize et ve girdi olarak hazırla
    girdiler = tokenizer(metin, return_tensors="pt", padding=True, truncation=True, max_length=256)
    girdiler = {key: val.to(device) for key, val in girdiler.items()}

    with torch.no_grad():
        çıktılar = model(**girdiler)

    logits = çıktılar.logits
    olasılıklar = torch.nn.functional.softmax(logits, dim=-1)[0]
    tahmin_edilen_sınıf = torch.argmax(logits, dim=-1).item()

    # Her kategori için olasılıkları biçimlendir
    kategori_olasılıkları = {
        kategori: f'{olasılıklar[idx].item():.2%}'
        for kategori, idx in kategori_haritasi.items()
    }

    # Sonuçları bir sözlük olarak hazırla ve geri döndür
    sonuc = {
        'Tahmin': ters_kategori_haritasi[tahmin_edilen_sınıf],
        'Tahmin Güveni': f'{olasılıklar[tahmin_edilen_sınıf].item():.2%}',
        'Kategori Olasılıkları': kategori_olasılıkları,
        'En Yüksek İkinci Tahmin': ters_kategori_haritasi[olasılıklar.argsort(descending=True)[1].item()],
        'İkinci Tahmin Olasılığı': f'{olasılıklar[olasılıklar.argsort(descending=True)[1]].item():.2%}',
    }

    return sonuc

if __name__ == "__main__":
    ornek_cumle = "sen beyin yerine ne kullanıyorsun tatlım?"
    analiz_sonucu = metni_analiz_et(ornek_cumle)
    print("\nAnaliz Sonuçları:")
    print(f"Analiz Edilen Cümle: {ornek_cumle}")
    print(f"Tahmin: {analiz_sonucu['Tahmin']}")
    print(f"Tahmin Güveni: {analiz_sonucu['Tahmin Güveni']}")
    print("--------------------------")
    print("Kategori Olasılıkları:")
    for kategori, olasilik in analiz_sonucu['Kategori Olasılıkları'].items():
        print(f"{kategori}: {olasilik}")
