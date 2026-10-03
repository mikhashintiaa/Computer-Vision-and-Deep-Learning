import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix


# =========================================================
# KONFIGURASI
# =========================================================
NUM_CLASSES = 3
BATCH_SIZE = 8

TEST_DIR = "test"
HASIL_DIR = "hasil"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CONFUSION MATRIX MODEL")
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

class_names = test_dataset.classes

print("\nClasses:")

for idx, cls in enumerate(class_names):
    print(f"{idx} = {cls}")

print("\nJumlah test :", len(test_dataset))


# =========================================================
# FUNGSI MODEL
# =========================================================
def load_resnet50():

    model = models.resnet50(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            os.path.join(
                HASIL_DIR,
                "resnet50_feature_best.pth"
            ),
            map_location=device
        )
    )

    return model


def load_resnet18():

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
            os.path.join(
                HASIL_DIR,
                "resnet18_feature_best.pth"
            ),
            map_location=device
        )
    )

    return model


def load_mobilenet():

    model = models.mobilenet_v3_small(
        weights=None
    )

    num_features = (
        model.classifier[3].in_features
    )

    model.classifier[3] = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            os.path.join(
                HASIL_DIR,
                "mobilenetv3_small_feature_best.pth"
            ),
            map_location=device
        )
    )

    return model


# =========================================================
# PREDIKSI
# =========================================================
def get_predictions(model):

    model = model.to(device)

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================
def plot_confusion_matrix(
    cm,
    title,
    cmap,
    filename
):

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=cmap
    )

    plt.title(
        title,
        fontsize=15,
        fontweight="bold"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=20,
        fontsize=10
    )

    plt.yticks(
        tick_marks,
        class_names,
        fontsize=10
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "True Label",
        fontsize=12
    )


    threshold = (
        cm.max() / 2.0
    )


    # isi angka matrix
    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                fontweight="bold",
                color=text_color
            )


    plt.tight_layout()

    output_path = os.path.join(
        HASIL_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Tersimpan : {output_path}"
    )


# =========================================================
# RESNET50
# =========================================================
print("\n" + "=" * 70)
print("RESNET50")
print("=" * 70)

resnet50 = load_resnet50()

labels, preds = get_predictions(
    resnet50
)

cm_resnet50 = confusion_matrix(
    labels,
    preds
)

print(cm_resnet50)

plot_confusion_matrix(
    cm_resnet50,
    "Confusion Matrix - ResNet50",
    "Blues",
    "confusion_matrix_resnet50.png"
)


# =========================================================
# RESNET18
# =========================================================
print("\n" + "=" * 70)
print("RESNET18")
print("=" * 70)

resnet18 = load_resnet18()

labels, preds = get_predictions(
    resnet18
)

cm_resnet18 = confusion_matrix(
    labels,
    preds
)

print(cm_resnet18)

plot_confusion_matrix(
    cm_resnet18,
    "Confusion Matrix - ResNet18",
    "Greens",
    "confusion_matrix_resnet18.png"
)


# =========================================================
# MOBILENETV3 SMALL
# =========================================================
print("\n" + "=" * 70)
print("MOBILENETV3-SMALL")
print("=" * 70)

mobilenet = load_mobilenet()

labels, preds = get_predictions(
    mobilenet
)

cm_mobilenet = confusion_matrix(
    labels,
    preds
)

print(cm_mobilenet)

plot_confusion_matrix(
    cm_mobilenet,
    "Confusion Matrix - MobileNetV3-Small",
    "Purples",
    "confusion_matrix_mobilenetv3_small.png"
)


print("\n" + "=" * 70)
print("SEMUA CONFUSION MATRIX SELESAI")
print("=" * 70)