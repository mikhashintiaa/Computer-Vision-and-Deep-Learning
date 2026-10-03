#  Transfer Learning - Balok-Classification

<p align="center">
  <b>Image Classification for Arm Robot Pick and Place Application</b>
</p>

---

## 👩‍💻 Project Identity

| Information | Detail |
|---|---|
| **Name** | Mikha Shintia Sitorus |
| **NIM** | 4222401060 |
| **Project** | Arm Robot – Pick and Place |
| **Task** | Image Classification / Transfer Learning |
| **Framework** | PyTorch |
| **Programming Language** | Python |
| **Models** | ResNet50, ResNet18, MobileNetV3-Small |
| **Number of Classes** | 3 |
| **Dataset Size** | 110 Images |

---

# 📌 Project Overview

This project implements an image classification system for an **Arm Robot Pick and Place** application.

The classification model is designed to identify three types of colored blocks:

- 🟢 `balok_hijaumuda`
- ⚫ `balok_hitam`
- 🌸 `balok_merahmuda`

Three Convolutional Neural Network architectures are evaluated:

1. **ResNet50**
2. **ResNet18**
3. **MobileNetV3-Small**

Each architecture is trained using three different training strategies:

- **Feature Extraction**
- **Partial Fine-Tuning**
- **Training from Scratch**

The purpose of this experiment is to compare the effect of different architectures and training methods based on:

- Validation Accuracy
- Test Accuracy
- Training Time
- Convergence Speed
- Inference Latency
- Approximate FPS
- Confusion Matrix

---

# 🎯 Project Objective

The main objective of this project is to determine an appropriate image classification architecture for an Arm Robot Pick and Place system.

The selected model should provide a good balance between:

**classification accuracy** and **inference speed**.

This is important because an arm robot requires reliable object recognition while still maintaining a sufficiently fast response during operation.

---

# 📊 Dataset

The dataset consists of **110 images** divided into three classes.

| Class | Number of Images |
|---|---:|
| balok_hijaumuda | 37 |
| balok_hitam | 37 |
| balok_merahmuda | 36 |
| **Total** | **110** |

The dataset was divided into:

| Dataset | Images |
|---|---:|
| Training | 77 |
| Validation | 22 |
| Testing | 11 |
| **Total** | **110** |

The split corresponds approximately to:

```text
Training   : 70%
Validation : 20%
Testing    : 10%
