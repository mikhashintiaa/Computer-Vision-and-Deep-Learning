import pandas as pd


# =========================================================
# DATA HASIL MOBILENETV3-SMALL
# =========================================================
data = {

    "Mode": [
        "Feature",
        "Partial",
        "Scratch"
    ],

    "Best Val Accuracy (%)": [
        72.73,
        81.82,
        31.82
    ],

    "Best Epoch": [
        1,
        2,
        1
    ],

    "Epoch Pertama >= 90%": [
        "Tidak tercapai",
        "Tidak tercapai",
        "Tidak tercapai"
    ],

    "Waktu Training (detik)": [
        32.42,
        26.15,
        50.71
    ],

    "Test Accuracy (%)": [
        81.82,
        63.64,
        36.36
    ]
}


# =========================================================
# DATAFRAME
# =========================================================
df = pd.DataFrame(data)


# =========================================================
# TAMPILKAN
# =========================================================
print("=" * 110)
print("REKAP HASIL MOBILENETV3-SMALL")
print("=" * 110)

print(
    df.to_string(index=False)
)


# =========================================================
# SIMPAN CSV
# =========================================================
output_file = (
    "hasil/rekap_mobilenetv3_small.csv"
)

df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 110)

print(
    f"Tersimpan : {output_file}"
)

print("=" * 110)