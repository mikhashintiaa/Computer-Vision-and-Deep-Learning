import time
import statistics

import torch
import torch.nn as nn
from torchvision import models


# =========================================================
# KONFIGURASI
# =========================================================
NUM_CLASSES = 3
MODEL_PATH = "hasil/resnet18_feature_best.pth"

WARMUP = 10
NUM_RUNS = 100


# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("LATENCY RESNET18 - FEATURE")
print("=" * 70)
print("Device :", device)


# =========================================================
# MODEL
# =========================================================
model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# DUMMY INPUT
# =========================================================
dummy_input = torch.randn(
    1,
    3,
    224,
    224
).to(device)


# =========================================================
# WARMUP
# =========================================================
print("\nWarmup...")

with torch.no_grad():

    for _ in range(WARMUP):
        _ = model(dummy_input)


if device.type == "cuda":
    torch.cuda.synchronize()


# =========================================================
# LATENCY
# =========================================================
latencies = []

print(
    f"Mengukur {NUM_RUNS} inference..."
)

with torch.no_grad():

    for _ in range(NUM_RUNS):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies.append(
            latency_ms
        )


# =========================================================
# HASIL
# =========================================================
avg_latency = statistics.mean(
    latencies
)

median_latency = statistics.median(
    latencies
)

min_latency = min(
    latencies
)

max_latency = max(
    latencies
)

fps = (
    1000 / avg_latency
)


print("\n" + "=" * 70)
print("HASIL LATENCY")
print("=" * 70)

print(
    "Model            : ResNet18 Feature"
)

print(
    f"Jumlah inference : {NUM_RUNS}"
)

print(
    f"Average latency  : {avg_latency:.2f} ms"
)

print(
    f"Median latency   : {median_latency:.2f} ms"
)

print(
    f"Minimum latency  : {min_latency:.2f} ms"
)

print(
    f"Maximum latency  : {max_latency:.2f} ms"
)

print(
    f"Approx FPS       : {fps:.2f}"
)

print("=" * 70)