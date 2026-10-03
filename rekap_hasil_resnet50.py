import pandas as pd

data = {
    "Mode": [
        "Feature",
        "Partial",
        "Scratch"
    ],

    "Best Val Accuracy (%)": [
        100.00,
        100.00,
        100.00
    ],

    "Best Epoch": [
        5,
        2,
        5
    ],

    "Epoch Pertama >= 90%": [
        2,
        2,
        3
    ],

    "Waktu Training (detik)": [
        212.43,
        231.06,
        439.40
    ]
}

df = pd.DataFrame(data)

print("=" * 80)
print("REKAP HASIL RESNET50")
print("=" * 80)

print(
    df.to_string(index=False)
)

df.to_csv(
    "hasil/rekap_resnet50.csv",
    index=False
)

print(
    "\nTersimpan: hasil/rekap_resnet50.csv"
)