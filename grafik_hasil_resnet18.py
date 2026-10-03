import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATH
# =========================================================
HASIL_DIR = "hasil"

feature_file = os.path.join(
    HASIL_DIR,
    "history_resnet18_feature.csv"
)

partial_file = os.path.join(
    HASIL_DIR,
    "history_resnet18_partial.csv"
)

scratch_file = os.path.join(
    HASIL_DIR,
    "history_resnet18_scratch.csv"
)


# =========================================================
# BACA DATA CSV
# =========================================================
feature = pd.read_csv(feature_file)
partial = pd.read_csv(partial_file)
scratch = pd.read_csv(scratch_file)


# =========================================================
# GRAFIK VALIDATION ACCURACY
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(
    feature["epoch"],
    feature["val_acc"],
    marker="o",
    label="Feature"
)

plt.plot(
    partial["epoch"],
    partial["val_acc"],
    marker="o",
    label="Partial"
)

plt.plot(
    scratch["epoch"],
    scratch["val_acc"],
    marker="o",
    label="Scratch"
)


# Garis target 90%
plt.axhline(
    y=90,
    linestyle="--",
    label="Target 90%"
)


# =========================================================
# LABEL
# =========================================================
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")

plt.title(
    "Perbandingan Validation Accuracy ResNet18"
)

plt.xticks(
    range(1, 11)
)

plt.ylim(
    0,
    105
)

plt.grid(True)

plt.legend()

plt.tight_layout()


# =========================================================
# SIMPAN
# =========================================================
output_path = os.path.join(
    HASIL_DIR,
    "grafik_val_accuracy_resnet18.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()


print("=" * 70)
print("GRAFIK RESNET18 SELESAI")
print("=" * 70)

print(
    f"Grafik tersimpan : {output_path}"
)