import pandas as pd


# =========================================================
# DATA HASIL RESNET18
# =========================================================
data = {

    "Mode": [
        "Feature",
        "Partial",
        "Scratch"
    ],

    "Best Val Accuracy (%)": [
        100.00,
        100.00,
        95.45
    ],

    "Best Epoch": [
        4,
        2,
        8
    ],

    "Epoch Pertama >= 90%": [
        3,
        1,
        6
    ],

    "Waktu Training (detik)": [
        80.40,
        103.24,
        171.78
    ],

    "Test Accuracy (%)": [
        100.00,
        100.00,
        100.00
    ]
}


# =========================================================
# DATAFRAME
# =========================================================
df = pd.DataFrame(data)


# =========================================================
# TAMPILKAN
# =========================================================
print("=" * 100)
print("REKAP HASIL RESNET18")
print("=" * 100)

print(
    df.to_string(index=False)
)


# =========================================================
# SIMPAN CSV
# =========================================================
output_file = "hasil/rekap_resnet18.csv"

df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 100)

print(
    f"Tersimpan : {output_file}"
)

print("=" * 100)