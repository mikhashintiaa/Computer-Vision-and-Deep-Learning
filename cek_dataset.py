import os
import hashlib
from collections import defaultdict

SPLITS = ["train", "val", "test"]

EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# =========================================================
# CEK JUMLAH DATA
# =========================================================
print("=" * 70)
print("CEK DATASET FINAL")
print("=" * 70)

grand_total = 0

for split in SPLITS:

    print(f"\n{split.upper()}")

    split_total = 0

    split_path = split

    if not os.path.exists(split_path):
        print("Folder tidak ditemukan.")
        continue

    classes = sorted([
        folder
        for folder in os.listdir(split_path)
        if os.path.isdir(
            os.path.join(split_path, folder)
        )
    ])

    for cls in classes:

        class_path = os.path.join(
            split_path,
            cls
        )

        images = [
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(EXTENSIONS)
        ]

        jumlah = len(images)

        split_total += jumlah

        print(
            f"{cls:20s} : {jumlah}"
        )

    grand_total += split_total

    print(
        f"TOTAL {split.upper():5s} : {split_total}"
    )


print("\n" + "=" * 70)
print(f"TOTAL SEMUA DATA : {grand_total}")
print("=" * 70)


# =========================================================
# CEK DUPLIKAT ANTAR SPLIT
# =========================================================
def file_hash(filepath):

    hasher = hashlib.md5()

    with open(filepath, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


hash_map = defaultdict(list)

for split in SPLITS:

    if not os.path.exists(split):
        continue

    for cls in os.listdir(split):

        class_path = os.path.join(
            split,
            cls
        )

        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):

            if not filename.lower().endswith(
                EXTENSIONS
            ):
                continue

            path = os.path.join(
                class_path,
                filename
            )

            hash_value = file_hash(path)

            hash_map[hash_value].append({
                "split": split,
                "class": cls,
                "path": path
            })


duplicate_groups = 0
leakage_groups = 0


for hash_value, files in hash_map.items():

    if len(files) > 1:

        duplicate_groups += 1

        splits_found = set(
            item["split"]
            for item in files
        )

        if len(splits_found) > 1:

            leakage_groups += 1

            print("\nWARNING - DUPLIKAT ANTAR SPLIT:")

            for item in files:

                print(
                    f"[{item['split']}] "
                    f"{item['path']}"
                )


print("\n" + "=" * 70)
print("HASIL CEK DUPLIKAT")
print("=" * 70)

print(
    "Kelompok duplikat     :",
    duplicate_groups
)

print(
    "Leakage antar split   :",
    leakage_groups
)

if leakage_groups == 0:

    print(
        "STATUS: AMAN - tidak ada file identik "
        "di train/val/test."
    )

else:

    print(
        "STATUS: PERLU DIPERBAIKI."
    )