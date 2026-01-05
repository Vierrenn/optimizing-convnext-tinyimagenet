# Optimizing ConvNeXt for TinyImageNet Classification

This repository presents an experimental study on improving **ConvNeXt-Tiny** performance for **TinyImageNet image classification** by integrating **self-attention mechanisms** and **aggressive regularization strategies**.
The project compares a baseline transfer learning setup with an improved fine-tuning configuration to address overfitting and enhance generalization on small-scale datasets.

---

## Overview

Convolutional Neural Networks (CNNs) remain a strong backbone for image classification tasks. ConvNeXt modernizes CNN architectures by adopting design principles inspired by Vision Transformers while retaining pure convolutional operations.

Despite its strong performance on large-scale datasets, ConvNeXt can suffer from overfitting when applied to smaller datasets such as TinyImageNet. This project investigates how **self-attention integration**, **full fine-tuning**, and **aggressive regularization** can improve ConvNeXt generalization in data-limited settings.

---

## Dataset

- **Dataset**: TinyImageNet
- **Number of classes**: 200
- **Image resolution**: 64 × 64 (resized to 224 × 224)
- **Task**: Multi-class image classification

Dataset reference:

- https://huggingface.co/datasets/zh-plus/tiny-imagenet

---

## Model Configurations

### Baseline Model

- ConvNeXt-Tiny pretrained on ImageNet-1K
- Frozen backbone
- Trainable classification head only
- Early stopping for regularization

### Improved Model

- Full fine-tuning of ConvNeXt-Tiny
- Lightweight self-attention module at the final stage
- Aggressive regularization:
  - RandAugment
  - Dropout
  - High weight decay
- Automatic Mixed Precision (AMP)

---

## Results

| Model             | Validation Accuracy |
| ----------------- | ------------------- |
| Baseline ConvNeXt | 82.61%              |
| Improved ConvNeXt | **86.30%**          |

Visualizations provided in the `results/` directory include:

- Learning curves (training vs validation accuracy and loss)
- Correct vs incorrect prediction samples
- Per-class sample predictions

---

## Project Structure

optimizing-convnext-tinyimagenet/
│
├── notebook/
│ ├── ConvNeXt_Baseline.ipynb
│ └── ConvNeXt_Improvement.ipynb
│
├── results/
│ ├── Baseline/
│ │ ├── baseline_training_curve.png
│ │ ├── baseline_8_correct_8_wrong.png
│ │ ├── baseline_per_class.png
│ │ └── baseline_validation_predictions.csv
│ │
│ └── Improvement/
│ ├── improvement_training_curve.png
│ ├── improvement_8_correct_8_wrong.png
│ ├── improvement_per_class.png
│ └── improvement_validation_predictions.csv
│
├── paper/
│ └── Final_Paper.pdf
│
├── presentation/
│ └── Final_Presentation.pdf
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Installation

git clone https://github.com/Vierrenn/optimizing-convnext-tinyimagenet.git
cd optimizing-convnext-tinyimagenet
pip install -r requirements.txt

---

## Usage

All experiments are provided as Jupyter notebooks.

1. Open the baseline experiment:
   notebook/ConvNeXt_Baseline.ipynb

2. Open the improved experiment:
   notebook/ConvNeXt_Improvement.ipynb

Each notebook includes:

- Dataset loading
- Model configuration
- Training and validation loops
- Evaluation and visualization

---

## Key Findings

- Freezing the backbone leads to faster convergence but limits generalization.
- Full fine-tuning improves performance on TinyImageNet when combined with strong regularization.
- Lightweight self-attention enhances feature representation at deeper stages.
- Aggressive regularization mitigates overfitting in data-limited settings.

---

## Notes

- This project is intended for academic and portfolio purposes.
- Results may vary depending on random seed and hardware configuration.
- Model checkpoints are not included due to size constraints.
