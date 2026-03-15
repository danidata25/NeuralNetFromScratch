# NeuralNetFromScratch 🧠

A fully manual implementation of a deep neural network (DNN) built from scratch using **NumPy and PyTorch tensors**, trained on the **MNIST** handwritten digit classification dataset.

---

## Overview

This project implements all core components of a deep neural network without using any high-level deep learning framework abstractions. Everything — forward pass, backward pass, weight updates — is written by hand.

Key features:
- **L-layer deep neural network** (configurable architecture)
- **Batch Normalization** support
- **L2 Regularization** support
- **Mini-batch gradient descent**
- **Early stopping** based on validation cost convergence
- Trained and evaluated on **MNIST** (60,000 training images, 10 classes)

---

## Architecture

```
Input (784) → [Linear → ReLU] × (L-1) → Linear → Softmax → Output (10)
```

Configurable via `layers_dims`, e.g.:
```python
layers_dims = [784, 20, 7, 5, 10]
```

---

## Implemented Components

| Module | Description |
|--------|-------------|
| `initialize_parameters` | He-style weight init |
| `linear_forward` | Z = WA + b |
| `relu` / `softmax` | Activation functions |
| `apply_batchnorm` | Batch normalization |
| `L_model_forward` | Full forward pass |
| `compute_cost` | Cross-entropy + optional L2 |
| `relu_backward` | ReLU gradient |
| `linear_backward` | Gradients w.r.t W, b, A |
| `L_model_backward` | Full backpropagation |
| `Update_parameters` | SGD with optional L2 weight decay |
| `L_layer_model` | Full training loop with mini-batches |
| `Predict` | Accuracy evaluation |

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook
```bash
jupyter notebook deep_learning_hw1.ipynb
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | [784, 20, 7, 5, 10] |
| Learning Rate | 0.009 |
| Epochs | 15 |
| Batch Size | 32 |
| Batch Norm | ✓ |
| L2 Reg (λ) | 0.07 |

---

## Dataset

**MNIST** — loaded via `tensorflow.keras.datasets.mnist`:
- 60,000 training images, 10,000 test images
- 28×28 grayscale images, flattened to 784
- Labels one-hot encoded to 10 classes

---

## Project Structure

```
NeuralNetFromScratch/
├── deep_learning_hw1.ipynb   # Main notebook
├── requirements.txt          # Dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```
