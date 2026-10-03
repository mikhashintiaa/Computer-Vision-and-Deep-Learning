import os
import time
import copy
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================================================
# KONFIGURASI
# =========================================================
MODE = "scratch"

NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_CLASSES = 3

TRAIN_DIR = "train"
VAL_DIR = "val"

OUTPUT_DIR = "hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRAINING RESNET18")
print("=" * 70)
print("Mode   :", MODE)
print("Device :", device)


# =========================================================
# TRANSFORM
# =========================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# DATASET
# =========================================================
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClasses:")
for idx, class_name in enumerate(train_dataset.classes):
    print(f"{idx} = {class_name}")

print("\nJumlah data:")
print("Train :", len(train_dataset))
print("Val   :", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# MODEL RESNET18 - SCRATCH
# =========================================================
model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(device)


# =========================================================
# INFO PARAMETER
# =========================================================
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nParameter trainable:")
print(
    f"{trainable_params:,} / {total_params:,}"
)

print("Learning Rate : 1e-3")


# =========================================================
# LOSS & OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# =========================================================
# TRAINING
# =========================================================
best_val_acc = 0.0
best_epoch = 0
first_90_epoch = None

best_model_wts = copy.deepcopy(
    model.state_dict()
)

history = []

start_training = time.time()


for epoch in range(1, NUM_EPOCHS + 1):

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch}/{NUM_EPOCHS}")
    print("=" * 70)

    # TRAIN
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        _, predictions = torch.max(
            outputs,
            1
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        running_total += labels.size(0)

    train_loss = (
        running_loss / running_total
    )

    train_acc = (
        100.0
        * running_correct
        / running_total
    )


    # VALIDATION
    model.eval()

    val_running_loss = 0.0
    val_running_correct = 0
    val_running_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            val_running_correct += (
                predictions == labels
            ).sum().item()

            val_running_total += (
                labels.size(0)
            )

    val_loss = (
        val_running_loss / val_running_total
    )

    val_acc = (
        100.0
        * val_running_correct
        / val_running_total
    )


    # CATAT EPOCH PERTAMA >=90%
    if (
        val_acc >= 90.0
        and first_90_epoch is None
    ):
        first_90_epoch = epoch


    # BEST MODEL
    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch

        best_model_wts = copy.deepcopy(
            model.state_dict()
        )

        print(">>> Best model diperbarui")


    # HISTORY
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")


# =========================================================
# SELESAI TRAINING
# =========================================================
end_training = time.time()

training_time = (
    end_training - start_training
)

model.load_state_dict(
    best_model_wts
)


# =========================================================
# SIMPAN MODEL
# =========================================================
model_path = os.path.join(
    OUTPUT_DIR,
    "resnet18_scratch_best.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# =========================================================
# SIMPAN HISTORY
# =========================================================
history_path = os.path.join(
    OUTPUT_DIR,
    "history_resnet18_scratch.csv"
)

with open(
    history_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


# =========================================================
# HASIL
# =========================================================
print("\n" + "=" * 70)
print("HASIL TRAINING")
print("=" * 70)

print("Model                 : ResNet18")
print(f"Mode                  : {MODE}")
print(f"Best Val Accuracy     : {best_val_acc:.2f}%")
print(f"Best Epoch            : {best_epoch}")

if first_90_epoch is not None:
    print(
        f"Epoch pertama >= 90%  : {first_90_epoch}"
    )
else:
    print(
        "Epoch pertama >= 90%  : Tidak tercapai"
    )

print(
    f"Waktu training        : {training_time:.2f} detik"
)

print(
    f"Model tersimpan       : {model_path}"
)

print(
    f"History tersimpan     : {history_path}"
)

print("=" * 70)