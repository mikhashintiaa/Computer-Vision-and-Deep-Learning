#  Transfer Learning - Balok-Classification

<p align="center">
  <b>Image Classification for Arm Robot Pick and Place Application</b>
</p>

---


<p align="center">
  <b>Perbandingan ResNet50, ResNet18, dan MobileNetV3-Small untuk Klasifikasi Balok pada Sistem Arm Robot Pick and Place</b>
</p>

---

## 👩‍💻 Identitas

| Informasi | Keterangan |
|---|---|
| **Nama** | Mikha Shintia Sitorus |
| **NIM** | 4222401060 |
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

Model klasifikasi digunakan untuk membedakan tiga jenis balok, yaitu:

- 🟢 `balok_hijaumuda`
- ⚫ `balok_hitam`
- 🌸 `balok_merahmuda`

Tiga arsitektur Convolutional Neural Network (CNN) dibandingkan dalam project ini:

1. **ResNet50**
2. **ResNet18**
3. **MobileNetV3-Small**

Setiap arsitektur diuji menggunakan tiga metode pelatihan:

- **Feature Extraction**
- **Partial Fine-Tuning**
- **Training from Scratch**

Evaluasi dilakukan berdasarkan:

- Akurasi validasi terbaik
- Epoch saat akurasi mencapai ≥ 90%
- Waktu pelatihan
- Akurasi data test
- Confusion Matrix
- Latensi inferensi
- Approximate FPS
- Stabilitas proses training

---

# 🎯 Tujuan Project

Tujuan utama project ini adalah menentukan model klasifikasi citra yang sesuai untuk sistem **Arm Robot Pick and Place**.

Model yang digunakan tidak hanya dituntut memiliki akurasi tinggi, tetapi juga harus memiliki waktu inferensi yang cukup cepat agar dapat digunakan pada sistem robot yang membutuhkan respons real-time.

Secara umum, project ini bertujuan untuk:

- Mengklasifikasikan balok berdasarkan kategori warna
- Membandingkan performa tiga arsitektur CNN
- Membandingkan tiga strategi transfer learning
- Mengetahui pengaruh pretrained ImageNet terhadap dataset kecil
- Mengukur kecepatan inferensi setiap model
- Menentukan keseimbangan antara akurasi dan kecepatan model

---

# 🧠 Arsitektur yang Digunakan

## 1. ResNet50

ResNet50 merupakan arsitektur CNN dengan 50 layer yang menggunakan konsep **Residual Connection**.

Residual Connection membantu mengurangi masalah vanishing gradient dan memungkinkan pembangunan jaringan yang lebih dalam.

ResNet50 memiliki jumlah parameter yang relatif besar sehingga memiliki kemampuan representasi fitur yang tinggi, namun membutuhkan sumber daya komputasi yang lebih besar.

---

## 2. ResNet18

ResNet18 menggunakan konsep residual yang sama dengan ResNet50, tetapi memiliki struktur yang lebih ringan.

Karena jumlah layer dan parameter lebih sedikit, ResNet18 memiliki:

- waktu training lebih singkat,
- inferensi lebih cepat,
- kebutuhan komputasi lebih rendah.

ResNet18 sangat menarik untuk sistem robot karena dapat memberikan keseimbangan antara akurasi dan kecepatan.

---

## 3. MobileNetV3-Small

MobileNetV3-Small dirancang khusus untuk perangkat dengan sumber daya terbatas.

Arsitektur ini menggunakan beberapa teknik efisiensi seperti:

- Depthwise Separable Convolution
- Squeeze-and-Excitation
- h-swish activation
- Arsitektur yang dioptimalkan untuk perangkat ringan

MobileNetV3-Small memiliki jumlah parameter jauh lebih sedikit dibandingkan ResNet18 dan ResNet50 sehingga memiliki kecepatan inferensi yang lebih tinggi.

---

# 📊 Dataset

Dataset terdiri dari **110 citra** yang dibagi ke dalam tiga kelas.

| Kelas | Jumlah Citra |
|---|---:|
| balok_hijaumuda | 37 |
| balok_hitam | 37 |
| balok_merahmuda | 36 |
| **Total** | **110** |

Dataset kemudian dibagi menjadi:

| Dataset | Jumlah |
|---|---:|
| Training | 77 |
| Validation | 22 |
| Testing | 11 |
| **Total** | **110** |

Pembagian dataset kurang lebih menggunakan proporsi:

```text
Training   : 70%
Validation : 20%
Testing    : 10%
