# 🎮 GTA San Andreas AI Sürüş Ajanı

> CNN tabanlı Taklit Öğrenme (Imitation Learning) ile GTA San Andreas Sürüş Okulu görevlerini tamamlayan otonom ajan.

---

## 📌 Proje Hakkında

Bu proje, bir insan oyuncunun oynarken ürettiği ekran görüntüsü ve klavye verilerini kullanarak sürüş kararları almayı öğrenen bir yapay zeka ajanı geliştirmeyi amaçlamaktadır. Model mimarisi **GTAI** (GTA AI) olarak adlandırılmıştır.

Ham piksel verisinden doğrudan aksiyon tahmini yapan bu yaklaşım, uçtan uca (end-to-end) öğrenme paradigmasının somut bir uygulamasıdır.

---

## 🧠 Nasıl Çalışır?

```
Ekran Görüntüsü (1280x720)
        │
        ▼
  Ön İşleme (224x224, normalize)
        │
        ▼
   GTAI CNN Modeli
   [Conv→BN→ReLU→Pool] x4
   → Global Avg Pool
   → FC 512 → FC 10
        │
        ▼
  Aksiyon Tahmini (10 sınıf)
  [W / A / S / D / W+A / W+D ...]
        │
        ▼
  Klavye Çıktısı (pydirectinput)
```

**Eğitim yöntemi:** Behavioral Cloning — sen oynarsın, GTAI kaydeder, sonra taklit eder.

---

## 📁 Proje Yapısı

```
gta-sa-ai/
│
├── capture/
│   ├── screen_capture.py     # MSS ile gerçek zamanlı ekran yakalama
│   ├── input_logger.py       # Klavye dinleme ve aksiyon etiketleme
│   └── collect_data.py       # Veri toplama ana scripti
│
├── models/
│   ├── cnn_model.py          # GTAI CNN mimarisi
│   └── dataset.py            # PyTorch Dataset ve DataLoader
│
├── train/
│   └── train.py              # Model eğitim döngüsü
│
├── play/
│   └── run_agent.py          # Eğitilmiş modeli oyunda çalıştırır
│
├── data/
│   └── sessions/             # Kayıt oturumları (otomatik oluşur)
│       └── <oturum_adı>/
│           ├── frames/       # .npy formatında frame'ler
│           ├── labels.npy    # Aksiyon etiketleri
│           └── meta.json     # Oturum bilgileri
│
├── checkpoints/              # Eğitilmiş model ağırlıkları (.pth)
├── requirements.txt
└── README.md
```

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.10+
- Windows 10/11 (pydirectinput Windows gerektirir)
- GTA San Andreas (PC)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/kullanici-adi/gta-sa-ai.git
cd gta-sa-ai

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

> ⚠️ PyTorch kurulumu için GPU'na uygun komutu [pytorch.org](https://pytorch.org/get-started/locally/) adresinden al. `torch` ve `torchvision` her zaman birlikte güncellenmeli.

---

## 🚀 Kullanım

### 1. Veri Toplama

GTA San Andreas'ı aç, Sürüş Okulu görevine gir. Ardından:

```bash
cd capture
python collect_data.py --session surus_okulu_01 --fps 30
```

- 3 saniyelik geri sayım sonrası kayıt başlar
- Oyuna geç ve görevi **başarıyla** tamamla
- Durdurmak için önizleme penceresinde `Q` veya `Ctrl+C`
- Her başarılı görev serisi için yeni bir oturum adı kullan

**Önerilen kayıt süresi:** 2 saat (≈ 216.000 frame)

```bash
# Birden fazla oturum örneği
python collect_data.py --session surus_okulu_02 --fps 30
python collect_data.py --session surus_okulu_03 --fps 30
```

---

### 2. Model Eğitimi

#### Lokal (GPU varsa)
```bash
cd train
python train.py --sessions surus_okulu_01 surus_okulu_02 --arch gtai --epochs 30
```

#### Google Colab (önerilen)
```python
# Colab'de çalıştır
from google.colab import drive
drive.mount('/content/drive')

# Veriyi Drive'dan kopyala
!cp -r /content/drive/MyDrive/gta-sa-ai /content/

# Eğitimi başlat
!python train/train.py \
    --sessions surus_okulu_01 surus_okulu_02 \
    --arch gtai \
    --epochs 50 \
    --batch 64
```

Eğitim tamamlandığında `checkpoints/best_gtai.pth` dosyası oluşur.

---

### 3. Ajanı Çalıştır

```bash
cd play
python run_agent.py --model ../checkpoints/best_gtai.pth --arch gtai
```

- Oyunu aç ve sürüş görevini başlat
- Script 5 saniye bekler, ardından kontrol GTAI'ya geçer
- Önizleme penceresinde aksiyon olasılıkları görünür
- Durdurmak için `Q` veya `Ctrl+C`

---

## 🎯 Aksiyon Uzayı

| Etiket | Eylem | Tuş(lar) |
|--------|-------|----------|
| 0 | Boşta | — |
| 1 | İleri | W |
| 2 | Sol | A |
| 3 | Sağ | D |
| 4 | Fren / Geri | S |
| 5 | İleri + Sol | W + A |
| 6 | İleri + Sağ | W + D |
| 7 | Fren + Sol | S + A |
| 8 | Fren + Sağ | S + D |
| 9 | El Freni | Space |

---

## 🏗️ GTAI Model Mimarisi

```
Girdi: (B, 3, 224, 224)

Conv2d(3→32)   + BatchNorm + ReLU + MaxPool   →  (B, 32,  112, 112)
Conv2d(32→64)  + BatchNorm + ReLU + MaxPool   →  (B, 64,   56,  56)
Conv2d(64→128) + BatchNorm + ReLU + MaxPool   →  (B, 128,  28,  28)
Conv2d(128→256)+ BatchNorm + ReLU + MaxPool   →  (B, 256,  14,  14)

Global Average Pooling                         →  (B, 256)
Linear(256→512) + ReLU + Dropout(0.4)         →  (B, 512)
Linear(512→10)                                 →  (B, 10)

Çıktı: 10 sınıf logit
```

---

## 📊 Eğitim Detayları

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| LR Scheduler | ReduceLROnPlateau |
| Batch Size | 32 |
| Dropout | 0.4 |
| Loss Fonksiyonu | CrossEntropyLoss |
| Sınıf Dengesi | WeightedRandomSampler |

---

## 🔄 Geliştirme Yol Haritası

- [x] Ekran yakalama altyapısı
- [x] Klavye kayıt ve etiketleme sistemi
- [x] GTAI CNN mimarisi
- [x] Eğitim pipeline (Imitation Learning)
- [x] Gerçek zamanlı inference
- [ ] Daha fazla oturum verisi toplama
- [ ] Model performans değerlendirmesi
- [ ] Reinforcement Learning entegrasyonu *(gelecek sürüm)*

---

## 📦 Bağımlılıklar

```
torch >= 2.1.0
torchvision >= 0.16.0
mss
pynput
pydirectinput
opencv-python
numpy >= 1.24.0
Pillow
```

---

## ⚠️ Önemli Notlar

- Veri toplarken yalnızca **başarıyla tamamlanan** görevleri kaydet
- Ani veya yanlış hareketler içeren bölümleri kaydetme
- `torch` ve `torchvision` sürümlerini her zaman birlikte güncelle
- `pydirectinput` yalnızca **Windows** üzerinde çalışır

---

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.
