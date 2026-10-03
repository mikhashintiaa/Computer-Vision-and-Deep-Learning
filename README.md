# Balok-Classification-Resnet18,50, Mobilenetv3



Tugas ini merupakan implementasi klasifikasi citra balok menggunakan tiga arsitektur deep learning:

- ResNet50
- ResNet18
- MobileNetV3-Small

Eksperimen dilakukan menggunakan tiga metode pelatihan:

- Feature Extraction
- Partial Fine-Tuning
- Training from Scratch

Dataset terdiri dari tiga kelas:
- `balok_hijaumuda`
- `balok_hitam`
- `balok_merahmuda`

Total dataset yang digunakan sebanyak **110 citra**.

---

## 📊 Dataset Split

| Split | Jumlah |
|---|---:|
| Train | 77 |
| Validation | 22 |
| Test | 11 |
| Total | 110 |

Pembagian dataset dilakukan secara terpisah untuk setiap kelas agar distribusi data tetap seimbang.

---

## 🧠 Arsitektur yang Digunakan

### ResNet50
Model convolutional neural network yang memiliki struktur residual dengan kedalaman lebih besar.

### ResNet18
Versi ResNet yang lebih ringan dengan jumlah parameter lebih sedikit sehingga lebih cepat dalam proses inferensi.

### MobileNetV3-Small
Model lightweight CNN yang dirancang untuk perangkat dengan keterbatasan komputasi dan kebutuhan real-time.

---

## 🔬 Metode Training

### 1. Feature Extraction
Bobot pretrained ImageNet digunakan dan hanya layer klasifikasi terakhir yang dilatih.

### 2. Partial Fine-Tuning
Sebagian layer akhir model dibuka untuk proses training bersama classifier.

### 3. Scratch
Semua bobot model diinisialisasi secara acak dan seluruh parameter dilatih dari awal.

---

## 📈 Hasil ResNet50

| Mode | Best Val Accuracy | Epoch ≥ 90% | Training Time | Test Accuracy |
|---|---:|---:|---:|---:|
| Feature | 100.00% | 2 | 212.43 s | 100.00% |
| Partial | 100.00% | 2 | 231.06 s | 90.91% |
| Scratch | 100.00% | 3 | 439.40 s | 100.00% |

---

## 📈 Hasil ResNet18

| Mode | Best Val Accuracy | Epoch ≥ 90% | Training Time | Test Accuracy |
|---|---:|---:|---:|---:|
| Feature | 100.00% | 3 | 80.40 s | 100.00% |
| Partial | 100.00% | 1 | 103.24 s | 100.00% |
| Scratch | 95.45% | 6 | 171.78 s | 100.00% |

---

## 📈 Hasil MobileNetV3-Small

| Mode | Best Val Accuracy | Epoch ≥ 90% | Training Time | Test Accuracy |
|---|---:|---:|---:|---:|
| Feature | 72.73% | Tidak tercapai | 32.42 s | 81.82% |
| Partial | 81.82% | Tidak tercapai | 26.15 s | 63.64% |
| Scratch | 31.82% | Tidak tercapai | 50.71 s | 36.36% |

---

## ⚡ Perbandingan Latency

Latency diuji pada CPU dengan 100 kali inference.

| Model | Mode | Avg Latency | Approx FPS |
|---|---|---:|---:|
| ResNet50 | Feature | 233.76 ms | 4.28 FPS |
| ResNet18 | Feature | 81.69 ms | 12.24 FPS |
| MobileNetV3-Small | Feature | 20.67 ms | 48.37 FPS |

---

## 📉 Grafik Validation Accuracy

### ResNet50

![ResNet50](hasil/grafik_val_accuracy.png)

### ResNet18

![ResNet18](hasil/grafik_val_accuracy_resnet18.png)

### MobileNetV3-Small

![MobileNetV3](hasil/grafik_val_accuracy_mobilenetv3_small.png)

---

## 📊 Perbandingan 3 Metode

![Comparison](hasil/grafik_perbandingan_3_metode.png)

---

## 🔢 Confusion Matrix

### ResNet50

![Confusion Matrix ResNet50](hasil/confusion_matrix_resnet50.png)

### ResNet18

![Confusion Matrix ResNet18](hasil/confusion_matrix_resnet18.png)

### MobileNetV3-Small

![Confusion Matrix MobileNetV3](hasil/confusion_matrix_mobilenetv3_small.png)

---

## 📌 Analisis 

ResNet50 dan ResNet18 menunjukkan performa klasifikasi yang sangat baik pada dataset yang digunakan. Keduanya mampu mencapai akurasi test hingga 100% pada beberapa metode training.

ResNet18 memberikan kompromi yang lebih baik antara akurasi dan kecepatan dibandingkan ResNet50. ResNet18 tetap mencapai test accuracy 100%, namun memiliki latency yang jauh lebih rendah.

MobileNetV3-Small memiliki keunggulan utama pada kecepatan inferensi. Model ini mencapai sekitar 48 FPS pada CPU, tetapi akurasi klasifikasinya lebih rendah dibandingkan kedua model ResNet.

Hasil eksperimen menunjukkan adanya trade-off antara kompleksitas model, akurasi, waktu training, dan latency.

---

## 🚀 Cara Menjalankan

Install dependencies:

```bash
pip install -r requirements.txt
