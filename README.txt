DATASET BALOK 110 CITRA

Kelas:
1. balok_hijautua = 37 citra
2. balok_hitam = 37 citra
3. balok_merahmuda = 36 citra
TOTAL = 110 citra

Pembagian yang sudah ditetapkan di metadata.csv:
- Train = 77 citra
- Validation = 22 citra
- Test = 11 citra

Distribusi per kelas:
- balok_hijautua: train 26, val 7, test 4
- balok_hitam: train 26, val 7, test 4
- balok_merahmuda: train 25, val 8, test 3

Catatan:
ZIP ini hanya menyimpan 110 file gambar asli di dataset_raw agar total citra tetap 110.
Jalankan: python split_dataset.py
untuk membuat folder train/val/test berdasarkan metadata.csv sebelum training.
