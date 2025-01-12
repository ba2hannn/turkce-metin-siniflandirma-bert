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