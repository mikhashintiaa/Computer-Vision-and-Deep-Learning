#  Transfer Learning - Balok-Classification


<p align="center">
  <b>Perbandingan ResNet50, ResNet18, dan MobileNetV3-Small untuk Klasifikasi Balok</b>
</p>

---


from pathlib import Path

## 👩‍💻 Identitas

| Informasi | Keterangan |
|---|---|
| **Nama** | Mikha Shintia Sitorus |
| **NIM** | 4222401060 |
| **Program Studi** | Teknologi Rekayasa Robotika |
| **Project** | Arm Robot |
| **Aplikasi** | Pick and Place |
| **Topik** | Klasifikasi Citra Balok |
| **Framework** | PyTorch |
| **Bahasa Pemrograman** | Python |
| **Model yang Diuji** | ResNet50, ResNet18, MobileNetV3-Small |
| **Jumlah Kelas** | 3 |
| **Jumlah Dataset** | 110 citra |
| **Tahun** | 2026 |

---

# 📌 Deskripsi Project

Project ini merupakan bagian dari pengembangan sistem **Arm Robot Pick and Place** yang menggunakan pendekatan **Computer Vision dan Deep Learning** untuk mengenali objek balok berdasarkan citra.

Tujuan utama eksperimen adalah membandingkan tiga arsitektur Convolutional Neural Network (CNN), yaitu:

- **ResNet50**
- **ResNet18**
- **MobileNetV3-Small**

Setiap arsitektur diuji menggunakan tiga strategi pelatihan:

1. **Feature Extraction**
2. **Partial Fine-Tuning**
3. **Training from Scratch**

Parameter utama yang dibandingkan meliputi:

- Best Validation Accuracy
- Epoch terbaik
- Epoch pertama ketika validation accuracy mencapai ≥ 90%
- Waktu training
- Test accuracy
- Confusion matrix
- Latency inferensi
- Approximate FPS

Eksperimen dilakukan pada perangkat **CPU** dengan input citra berukuran **224 × 224 piksel**.

---

# 🧱 Kelas Dataset

Dataset terdiri dari tiga kelas balok:

| Label | Nama Kelas |
|---:|---|
| 0 | `balok_hijaumuda` |
| 1 | `balok_hitam` |
| 2 | `balok_merahmuda` |

Total dataset adalah **110 citra**.

Distribusi awal per kelas:

| Kelas | Jumlah Citra |
|---|---:|
| balok_hijaumuda | 37 |
| balok_hitam | 37 |
| balok_merahmuda | 36 |
| **Total** | **110** |

---

# 📊 Pembagian Dataset

Dataset dibagi menjadi data training, validation, dan testing.

| Kelas | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| balok_hijaumuda | 26 | 7 | 4 | 37 |
| balok_hitam | 26 | 7 | 4 | 37 |
| balok_merahmuda | 25 | 8 | 3 | 36 |
| **Total** | **77** | **22** | **11** | **110** |

Proporsi keseluruhan:

- **Train:** 77 citra
- **Validation:** 22 citra
- **Test:** 11 citra

Pemeriksaan dataset menunjukkan:

- Kelompok file duplikat: **0**
- Data leakage antar split: **0**
- Status dataset: **AMAN**

> Catatan: test set hanya berisi 11 citra, sehingga satu kesalahan prediksi akan mengubah akurasi sekitar 9,09%. Hasil test karena itu perlu dibaca bersama validation curve dan confusion matrix.

---

## 🗂️ Struktur Folder

```text
dataset_balokk/
│
├── dataset_raw/
│   ├── balok_hijaumuda/
│   ├── balok_hitam/
│   └── balok_merahmuda/
│
├── train/
│   ├── balok_hijaumuda/
│   ├── balok_hitam/
│   └── balok_merahmuda/
│
├── val/
│   ├── balok_hijaumuda/
│   ├── balok_hitam/
│   └── balok_merahmuda/
│
├── test/
│   ├── balok_hijaumuda/
│   ├── balok_hitam/
│   └── balok_merahmuda/
│
├── hasil/
│   ├── grafik_val_accuracy_resnet50.png
│   ├── grafik_val_accuracy_resnet18.png
│   ├── grafik_val_accuracy_mobilenetv3_small.png
│   ├── grafik_perbandingan_3_metode.png
│   ├── confusion_matrix_resnet50.png
│   ├── confusion_matrix_resnet18.png
│   └── confusion_matrix_mobilenetv3_small.png
│
├── metadata.csv
└── README.md
---

> Jika ukuran file `.pth` terlalu besar untuk GitHub biasa, model dapat disimpan menggunakan **Git LFS** atau tidak diunggah dan hanya history/hasil evaluasi yang disertakan.

---

# ⚙️ Konfigurasi Training

Konfigurasi utama yang digunakan:

| Parameter | Nilai |
|---|---|
| Input Size | 224 × 224 |
| Batch Size | 8 |
| Epoch | 10 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Device | CPU |
| Jumlah Kelas | 3 |

Normalisasi menggunakan statistik ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Augmentasi data training:

```python
transforms.RandomResizedCrop(224)
transforms.RandomHorizontalFlip()
transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)
```

---

# 🧪 Metode Training

## 1. Feature Extraction

Model menggunakan bobot awal **ImageNet**.

Feature extractor dibekukan dan hanya bagian classifier terakhir yang dilatih.

Learning rate:

```text
Classifier = 1e-3
```

Keuntungan:

- Waktu training singkat
- Cocok untuk dataset kecil
- Memanfaatkan representasi visual yang telah dipelajari dari ImageNet

---

## 2. Partial Fine-Tuning

Model menggunakan bobot awal **ImageNet**, tetapi bagian akhir feature extractor dibuka untuk ikut dilatih.

Untuk ResNet:

```text
layer4 = 1e-4
fc     = 1e-3
```

Untuk MobileNetV3-Small:

```text
feature block terakhir = 1e-4
classifier terakhir    = 1e-3
```

Tujuannya agar fitur tingkat tinggi dapat menyesuaikan diri terhadap karakteristik dataset balok.

---

## 3. Training from Scratch

Model diinisialisasi menggunakan bobot acak:

```python
weights=None
```

Seluruh parameter dilatih dengan:

```text
Learning Rate = 1e-3
```

Mode ini tidak memanfaatkan ImageNet sehingga membutuhkan proses pembelajaran fitur dari awal.

---

# 📈 Hasil ResNet50

## Tabel Hasil

| Mode | Best Val Accuracy | Best Epoch | Epoch ≥90% | Training Time | Test Accuracy |
|---|---:|---:|---:|---:|---:|
| Feature | **100.00%** | 5 | 2 | **212.43 s** | **100.00%** |
| Partial | **100.00%** | 2 | 2 | 231.06 s | 90.91% |
| Scratch | **100.00%** | 5 | 3 | 439.40 s | **100.00%** |

## Grafik Validation Accuracy

<p align="center">
  <img src="hasil/grafik_val_accuracy.png" width="760">
</p>

## Confusion Matrix ResNet50

<p align="center">
  <img src="hasil/confusion_matrix_resnet50.png" width="600">
</p>

### Analisis ResNet50

ResNet50 menunjukkan performa validation yang sangat tinggi pada ketiga metode. Feature extraction mencapai 100% validation accuracy pada epoch ke-5 dan sudah melewati 90% sejak epoch ke-2. Partial fine-tuning mencapai 100% lebih cepat, yaitu pada epoch ke-2, tetapi test accuracy turun menjadi 90,91%.

Scratch juga mampu mencapai 100% validation dan 100% test accuracy, tetapi waktu training mencapai 439,40 detik, lebih dari dua kali waktu feature extraction. Selain itu, kurva validation scratch lebih fluktuatif.

Untuk implementasi ResNet50 pada eksperimen ini, **feature extraction dipilih sebagai model representatif** karena memberikan test accuracy 100% dengan waktu training lebih efisien dibandingkan scratch.

---

# 📈 Hasil ResNet18

## Tabel Hasil

| Mode | Best Val Accuracy | Best Epoch | Epoch ≥90% | Training Time | Test Accuracy |
|---|---:|---:|---:|---:|---:|
| Feature | **100.00%** | 4 | 3 | **80.40 s** | **100.00%** |
| Partial | **100.00%** | 2 | **1** | 103.24 s | **100.00%** |
| Scratch | 95.45% | 8 | 6 | 171.78 s | **100.00%** |

## Grafik Validation Accuracy

<p align="center">
  <img src="hasil/grafik_val_accuracy_resnet18.png" width="760">
</p>

## Confusion Matrix ResNet18

<p align="center">
  <img src="hasil/confusion_matrix_resnet18.png" width="600">
</p>

### Analisis ResNet18

ResNet18 memberikan hasil yang sangat baik pada feature dan partial fine-tuning. Partial mencapai validation accuracy ≥90% sejak epoch pertama dan mencapai 100% pada epoch kedua. Feature mencapai 100% pada epoch ke-4.

Scratch memiliki pola yang lebih tidak stabil dan best validation accuracy hanya 95,45%, tetapi model terbaiknya tetap mengklasifikasikan seluruh 11 citra test dengan benar.

Dari sisi efisiensi, feature extraction membutuhkan waktu hanya 80,40 detik dan tetap memperoleh 100% validation serta 100% test accuracy. Hal ini menunjukkan bahwa ResNet18 mampu mempertahankan performa tinggi dengan beban komputasi lebih rendah daripada ResNet50.

---

# 📈 Hasil MobileNetV3-Small

## Tabel Hasil

| Mode | Best Val Accuracy | Best Epoch | Epoch ≥90% | Training Time | Test Accuracy |
|---|---:|---:|---|---:|---:|
| Feature | **72.73%** | 1 | Tidak tercapai | 32.42 s | **81.82%** |
| Partial | **81.82%** | 2 | Tidak tercapai | **26.15 s** | 63.64% |
| Scratch | 31.82% | 1 | Tidak tercapai | 50.71 s | 36.36% |

## Grafik Validation Accuracy

<p align="center">
  <img src="hasil/grafik_val_accuracy_mobilenetv3_small.png" width="760">
</p>

## Confusion Matrix MobileNetV3-Small

<p align="center">
  <img src="hasil/confusion_matrix_mobilenetv3_small.png" width="600">
</p>

### Analisis MobileNetV3-Small

MobileNetV3-Small memiliki waktu training paling rendah di antara ketiga arsitektur, tetapi hasil akurasinya berada di bawah ResNet.

Mode feature menghasilkan test accuracy terbaik sebesar 81,82% dengan 9 prediksi benar dari 11 citra test. Berdasarkan evaluasi per kelas, model feature mengenali seluruh citra `balok_hitam`, tetapi masih melakukan kesalahan pada `balok_hijaumuda` dan `balok_merahmuda`.

Partial fine-tuning menghasilkan validation accuracy lebih tinggi daripada feature, yaitu 81,82%, tetapi test accuracy turun menjadi 63,64%. Pada test set, seluruh citra `balok_hijaumuda` gagal diklasifikasikan dengan benar.

Scratch menghasilkan validation accuracy 31,82% dan test accuracy 36,36%. Nilai tersebut menunjukkan bahwa training dari nol pada dataset yang sangat kecil tidak cukup untuk menghasilkan generalisasi yang baik pada MobileNetV3-Small.

---

# 📊 Perbandingan Tiga Metode pada Semua Arsitektur

<p align="center">
  <img src="hasil/grafik_perbandingan_3_metode.png" width="850">
</p>

## Best Validation Accuracy

| Arsitektur | Feature | Partial | Scratch |
|---|---:|---:|---:|
| ResNet50 | **100.00%** | **100.00%** | **100.00%** |
| ResNet18 | **100.00%** | **100.00%** | 95.45% |
| MobileNetV3-Small | 72.73% | **81.82%** | 31.82% |

Grafik menunjukkan bahwa penggunaan pretrained ImageNet memberikan keuntungan yang besar terutama pada dataset kecil. ResNet50 dan ResNet18 mampu mencapai performa validation sangat tinggi ketika menggunakan feature extraction maupun partial fine-tuning.

---

# 📋 Rekap 9 Eksperimen

| Arsitektur | Mode | Best Val Acc | Best Epoch | Epoch ≥90% | Training Time | Test Acc |
|---|---|---:|---:|---|---:|---:|
| ResNet50 | Feature | 100.00% | 5 | 2 | 212.43 s | 100.00% |
| ResNet50 | Partial | 100.00% | 2 | 2 | 231.06 s | 90.91% |
| ResNet50 | Scratch | 100.00% | 5 | 3 | 439.40 s | 100.00% |
| ResNet18 | Feature | 100.00% | 4 | 3 | 80.40 s | 100.00% |
| ResNet18 | Partial | 100.00% | 2 | 1 | 103.24 s | 100.00% |
| ResNet18 | Scratch | 95.45% | 8 | 6 | 171.78 s | 100.00% |
| MobileNetV3-Small | Feature | 72.73% | 1 | Tidak tercapai | 32.42 s | 81.82% |
| MobileNetV3-Small | Partial | 81.82% | 2 | Tidak tercapai | 26.15 s | 63.64% |
| MobileNetV3-Small | Scratch | 31.82% | 1 | Tidak tercapai | 50.71 s | 36.36% |

---

# ⚡ Latency Model Terpilih

Latency diukur menggunakan:

- Batch size = 1
- Input = `1 × 3 × 224 × 224`
- Warmup = 10 inference
- Pengukuran = 100 inference
- Device = CPU

| Model | Mode | Test Accuracy | Average Latency | Median Latency | Approx FPS |
|---|---|---:|---:|---:|---:|
| ResNet50 | Feature | 100.00% | 233.76 ms | 228.23 ms | 4.28 |
| ResNet18 | Feature | **100.00%** | **81.69 ms** | **81.16 ms** | **12.24** |
| MobileNetV3-Small | Feature | 81.82% | **20.67 ms** | **20.36 ms** | **48.37** |

---

# 🔍 Analisis Perbandingan Arsitektur

## ResNet50

ResNet50 memiliki kapasitas model paling besar di antara tiga arsitektur yang diuji. Model ini memberikan akurasi sangat tinggi, tetapi membutuhkan waktu training dan latency inferensi paling besar.

Pada CPU, latency rata-rata mencapai **233,76 ms**, sehingga throughput hanya sekitar **4,28 FPS**.

ResNet50 sesuai ketika akurasi menjadi prioritas dan tersedia sumber daya komputasi yang cukup.

---

## ResNet18

ResNet18 memberikan keseimbangan yang sangat baik antara akurasi dan efisiensi.

Feature extraction menghasilkan:

- Validation accuracy: **100%**
- Test accuracy: **100%**
- Training time: **80,40 detik**
- Average latency: **81,69 ms**
- Approx FPS: **12,24 FPS**

Dibandingkan ResNet50, ResNet18 mempertahankan akurasi yang sama pada test set tetapi mempunyai latency CPU yang jauh lebih rendah.

---

## MobileNetV3-Small

MobileNetV3-Small merupakan arsitektur paling ringan dan memiliki latency paling rendah.

Feature extraction menghasilkan:

- Test accuracy: **81,82%**
- Average latency: **20,67 ms**
- Approx FPS: **48,37 FPS**

Model ini sangat cepat, tetapi akurasinya belum menyamai ResNet18 maupun ResNet50 pada dataset ini.

Dengan dataset yang lebih banyak dan tuning yang lebih lanjut, MobileNetV3-Small berpotensi menjadi pilihan untuk sistem embedded atau real-time.

---

# 🏁 Kesimpulan

Eksperimen menunjukkan bahwa strategi transfer learning memberikan manfaat besar pada dataset dengan jumlah citra terbatas.

Beberapa temuan utama:

1. **ResNet50** memberikan performa akurasi sangat tinggi tetapi memiliki latency dan waktu training paling besar.
2. **ResNet18** mampu memberikan validation dan test accuracy 100% dengan waktu training dan latency yang jauh lebih rendah dibandingkan ResNet50.
3. **MobileNetV3-Small** memberikan inferensi paling cepat, tetapi akurasinya masih lebih rendah pada dataset ini.
4. Training from scratch cenderung lebih tidak stabil karena hanya tersedia 77 citra training.
5. Pretrained ImageNet sangat membantu proses konvergensi pada dataset kecil.

Untuk sistem **Arm Robot Pick and Place** pada eksperimen ini, **ResNet18 Feature Extraction** merupakan kandidat yang sangat kuat karena mempertahankan test accuracy 100% dengan latency CPU yang jauh lebih rendah daripada ResNet50.

MobileNetV3-Small tetap menarik jika kebutuhan utama sistem adalah kecepatan inferensi dan keterbatasan perangkat keras.

---

# ▶️ Cara Menjalankan Project

## 1. Aktifkan Virtual Environment

Windows PowerShell:

```powershell
..\.venv\Scripts\Activate.ps1
```

---

## 2. Install Dependency

```powershell
pip install torch torchvision pandas matplotlib scikit-learn pillow
```

---

## 3. Split Dataset

```powershell
python split_dataset.py
```

---

## 4. Cek Dataset

```powershell
python cek_dataset.py
```

---

## 5. Training Model

Contoh:

```powershell
python train_resnet50.py
python train_resnet18.py
python train_mobilenetv3_feature.py
```

Mode pada masing-masing script dapat diatur menjadi:

```python
MODE = "feature"
MODE = "partial"
MODE = "scratch"
```

---

## 6. Test Model

```powershell
python test_resnet50.py
python test_resnet18.py
python test_mobilenetv3.py
```

---

## 7. Latency

```powershell
python latency_resnet50.py
python latency_resnet18.py
python latency_mobilenetv3.py
```

---

## 8. Grafik

```powershell
python grafik_hasil.py
python grafik_resnet18.py
python grafik_mobilenetv3.py
python grafik_perbandingan_3_metode.py
```

---

## 9. Confusion Matrix

```powershell
python confusion_matrix_models.py
```

---

# 📁 Output Utama

Setelah eksperimen selesai, beberapa file utama yang dapat digunakan dalam laporan adalah:

```text
hasil/
├── grafik_val_accuracy.png
├── grafik_val_accuracy_resnet18.png
├── grafik_val_accuracy_mobilenetv3_small.png
├── grafik_perbandingan_3_metode.png
├── confusion_matrix_resnet50.png
├── confusion_matrix_resnet18.png
├── confusion_matrix_mobilenetv3_small.png
├── rekap_resnet50.csv
├── rekap_resnet18.csv
└── rekap_mobilenetv3_small.csv
```

---

# 📝 Catatan Eksperimen

Dataset pada project ini relatif kecil, yaitu **110 citra**. Oleh karena itu, hasil akurasi yang sangat tinggi perlu dibaca dengan mempertimbangkan ukuran validation dan test set.

Test set hanya memiliki 11 citra sehingga hasil test belum dapat dianggap sebagai estimasi performa yang sangat kuat untuk kondisi dunia nyata.

Pengembangan selanjutnya dapat dilakukan dengan:

- Menambah jumlah citra per kelas
- Menambah variasi pencahayaan
- Menambah variasi posisi dan rotasi balok
- Menggunakan latar belakang yang lebih beragam
- Menguji model langsung dari kamera robot
- Mengukur latency end-to-end termasuk preprocessing citra
- Menguji model pada perangkat target robot
- Menggunakan cross-validation untuk evaluasi yang lebih kuat

---

## 📌 Ringkasan Akhir

| Model Terpilih | Test Accuracy | Avg Latency | FPS |
|---|---:|---:|---:|
| ResNet50 Feature | 100.00% | 233.76 ms | 4.28 |
| ResNet18 Feature | **100.00%** | **81.69 ms** | **12.24** |
| MobileNetV3-Small Feature | 81.82% | 20.67 ms | 48.37 |

**Project:** Arm Robot Pick and Place  
**Task:** Klasifikasi Balok dengan Transfer Learning  
**Author:** Mikha Shintia Sitorus — 4222401060
'''

path = Path("/mnt/data/README_GitHub_Klasifikasi_Balok.md")
path.write_text(readme, encoding="utf-8")
print(path)
