import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================================================
# KONFIGURASI
# =========================================================
NUM_CLASSES = 3
BATCH_SIZE = 8
TEST_DIR = "test"
HASIL_DIR = "hasil"


# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TEST RESNET50")
print("=" * 70)
print("Device :", device)


# =========================================================
# TRANSFORM TEST
# =========================================================
test_transform = transforms.Compose([
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
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
for idx, class_name in enumerate(test_dataset.classes):
    print(f"{idx} = {class_name}")

print("\nJumlah test :", len(test_dataset))


# =========================================================
# FUNGSI MEMBUAT MODEL
# =========================================================
def create_model():

    model = models.resnet50(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model


# =========================================================
# FUNGSI TEST
# =========================================================
def evaluate_model(mode):

    print("\n" + "=" * 70)
    print(f"TEST MODE : {mode.upper()}")
    print("=" * 70)

    model = create_model()

    model_path = os.path.join(
        HASIL_DIR,
        f"resnet50_{mode}_best.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    correct = 0
    total = 0

    class_correct = [
        0 for _ in test_dataset.classes
    ]

    class_total = [
        0 for _ in test_dataset.classes
    ]

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            for label, prediction in zip(
                labels,
                predictions
            ):

                label_index = label.item()

                class_total[
                    label_index
                ] += 1

                if label == prediction:

                    class_correct[
                        label_index
                    ] += 1


    accuracy = (
        100.0 * correct / total
    )

    print(
        f"Test Accuracy : {accuracy:.2f}%"
    )

    print(
        f"Benar         : {correct}/{total}"
    )


    print("\nAccuracy per kelas:")

    for idx, class_name in enumerate(
        test_dataset.classes
    ):

        if class_total[idx] > 0:

            class_accuracy = (
                100.0
                * class_correct[idx]
                / class_total[idx]
            )

        else:
            class_accuracy = 0.0

        print(
            f"{class_name:20s} : "
            f"{class_accuracy:.2f}% "
            f"({class_correct[idx]}/"
            f"{class_total[idx]})"
        )


    return accuracy


# =========================================================
# TEST 3 MODEL
# =========================================================
results = {}

for mode in [
    "feature",
    "partial",
    "scratch"
]:

    results[mode] = evaluate_model(
        mode
    )


# =========================================================
# RINGKASAN
# =========================================================
print("\n" + "=" * 70)
print("RINGKASAN TEST RESNET50")
print("=" * 70)

for mode, accuracy in results.items():

    print(
        f"{mode:10s} : "
        f"{accuracy:.2f}%"
    )

print("=" * 70)