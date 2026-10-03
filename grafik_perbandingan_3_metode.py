import os
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# DATA HASIL EKSPERIMEN
# =========================================================
metode = [
    "Feature",
    "Partial",
    "Scratch"
]

# Best Validation Accuracy masing-masing metode
resnet50 = [
    100.00,
    100.00,
    100.00
]

resnet18 = [
    100.00,
    100.00,
    95.45
]

mobilenetv3 = [
    72.73,
    81.82,
    31.82
]


# =========================================================
# POSISI BAR
# =========================================================
x = np.arange(len(metode))

width = 0.25


# =========================================================
# BUAT GRAFIK
# =========================================================
plt.figure(
    figsize=(10, 6)
)


bar1 = plt.bar(
    x - width,
    resnet50,
    width,
    label="ResNet50"
)

bar2 = plt.bar(
    x,
    resnet18,
    width,
    label="ResNet18"
)

bar3 = plt.bar(
    x + width,
    mobilenetv3,
    width,
    label="MobileNetV3-Small"
)


# =========================================================
# GARIS TARGET 90%
# =========================================================
plt.axhline(
    y=90,
    linestyle="--",
    label="Target 90%"
)


# =========================================================
# LABEL
# =========================================================
plt.xlabel(
    "Metode Training"
)

plt.ylabel(
    "Best Validation Accuracy (%)"
)

plt.title(
    "Perbandingan Tiga Metode Training pada "
    "ResNet50, ResNet18, dan MobileNetV3-Small"
)


plt.xticks(
    x,
    metode
)

plt.ylim(
    0,
    110
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)


# =========================================================
# TAMPILKAN NILAI DI ATAS BAR
# =========================================================
def tambah_label(bars):

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )


tambah_label(bar1)
tambah_label(bar2)
tambah_label(bar3)


# =========================================================
# SIMPAN
# =========================================================
plt.tight_layout()

OUTPUT_DIR = "hasil"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_path = os.path.join(
    OUTPUT_DIR,
    "grafik_perbandingan_3_metode.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()


print("=" * 70)
print("GRAFIK PERBANDINGAN 3 METODE SELESAI")
print("=" * 70)

print(
    f"Grafik tersimpan : {output_path}"
)