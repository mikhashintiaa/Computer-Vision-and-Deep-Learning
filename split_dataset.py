import os
import csv
import random
import shutil
from pathlib import Path


# =========================================================
# KONFIGURASI
# =========================================================
SEED = 42

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "dataset_raw"

TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

METADATA_PATH = BASE_DIR / "metadata.csv"

random.seed(SEED)


# =========================================================
# HAPUS SPLIT LAMA JIKA ADA
# =========================================================
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# AMBIL DAFTAR KELAS OTOMATIS
# =========================================================
classes = sorted([
    folder.name
    for folder in RAW_DIR.iterdir()
    if folder.is_dir()
])

print("=" * 70)
print("SPLIT DATASET")
print("=" * 70)

print("\nKelas ditemukan:")

for cls in classes:
    print("-", cls)


# =========================================================
# PROSES SPLIT
# =========================================================
metadata = []

total_train = 0
total_val = 0
total_test = 0

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


for cls in classes:

    class_dir = RAW_DIR / cls

    images = [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in extensions
    ]

    images = sorted(images)

    random.shuffle(images)

    jumlah = len(images)

    # =====================================================
    # PEMBAGIAN KHUSUS DATASET 110 GAMBAR
    # =====================================================
    if jumlah == 37:

        train_count = 26
        val_count = 7
        test_count = 4

    elif jumlah == 36:

        train_count = 25
        val_count = 8
        test_count = 3

    else:

        raise ValueError(
            f"Kelas '{cls}' memiliki {jumlah} gambar. "
            f"Seharusnya 36 atau 37 gambar."
        )


    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]


    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }


    # =====================================================
    # COPY FILE
    # =====================================================
    for split_name, files in split_data.items():

        destination = (
            BASE_DIR /
            split_name /
            cls
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for src in files:

            dst = destination / src.name

            shutil.copy2(
                src,
                dst
            )

            metadata.append({
                "filename": src.name,
                "class": cls,
                "split": split_name
            })


    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)


    print(f"\nKelas : {cls}")
    print(f"Total : {jumlah}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")


# =========================================================
# BUAT METADATA.CSV BARU
# =========================================================
with open(
    METADATA_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "class",
            "split"
        ]
    )

    writer.writeheader()

    writer.writerows(metadata)


# =========================================================
# RINGKASAN
# =========================================================
print("\n" + "=" * 70)
print("HASIL SPLIT")
print("=" * 70)

print(
    f"Train : {total_train}"
)

print(
    f"Val   : {total_val}"
)

print(
    f"Test  : {total_test}"
)

print("-" * 30)

print(
    f"TOTAL : "
    f"{total_train + total_val + total_test}"
)

print(
    f"\nMetadata tersimpan:"
)

print(
    METADATA_PATH
)

print("=" * 70)