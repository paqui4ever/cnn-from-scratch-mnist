
<div align="center">

# CNN from Scratch: MNIST Digit Classification

**Convolutional Neural Network from Scratch for MNIST Digit Classification**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DirectML](https://img.shields.io/badge/DirectML-0078D4?style=flat&logo=microsoft&logoColor=white)](https://github.com/microsoft/DirectML)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


</div>

## 📋 Overview

This project implements a convolutional neural network (CNN) from scratch using PyTorch to gain a deeper understanding of CNN design choices and training dynamics. The model achieves **98.42% accuracy**  on the MNIST test set using a simple yet effective architecture with two convolutional layers and fully connected classifiers.

## 📁 Project Structure

```
cnn-from-scratch-mnist/
├── 📂 data/
│   ├── __init__.py
│   └── dataset.py             # MNIST data loading (no augmentation)
├── 📂 models/
│   ├── __init__.py
│   └── cnn.py                 # SimpleCNN architecture
├── 📂 notebooks/
│   └── MNIST.ipynb            # Original experimental notebook
├── 📂 checkpoints/            # Saved model checkpoints (gitignored)
├── 📂 results/                # Evaluation outputs (gitignored)
├── 🔧 train.py                # Training script
├── 📈 evaluate.py             # Evaluation script
├── 📋 requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Installation

### Setup

**Clone the repository:**
   ```bash
   git clone https://github.com/paqui4ever/cnn-from-scratch-mnist.git
   cd cnn-from-scratch-mnist
   ```

**Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 🏋️ Training

Train the model with default parameters (100 epochs, lr=0.001, batch_size=32):

```bash
python train.py
```

**⚙️ Training Options:**

```bash
python train.py --epochs 100 --lr 0.001 --batch-size 32 --device directml
```

**📝 Arguments:**
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--device`: Device to use - `cuda`, `cpu`, or `directml` (auto-detect if not specified)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--resume`: Path to checkpoint to resume training
- `--save-every`: Save checkpoint every N epochs (default: 10)

**⚡ Example - Quick Training (5 epochs):**
```bash
python train.py --epochs 5 --batch-size 64
```

### 📊 Evaluation

Evaluate a trained model and generate visualizations:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

**⚙️ Evaluation Options:**

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --num-samples 16 --output-dir results
```

**📝 Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--batch-size`: Batch size (default: 32)
- `--device`: Device to use (auto-detect if not specified)
- `--output-dir`: Directory to save results (default: `results/`)
- `--num-samples`: Number of samples to visualize (default: 16)
- `--show-plots`: Display plots instead of saving

**📸 Generated Outputs:**
- `sample_predictions.png` - Random sample predictions
- `confusion_matrix.png` - Confusion matrix heatmap
- `misclassified.png` - Examples of misclassified digits

## 🏗️ Architecture

The SimpleCNN model uses a straightforward architecture that balances simplicity with expressiveness:

### 💡 Design Philosophy

- **Strided Convolution**: Uses stride=2 in the second convolutional layer for spatial downsampling instead of max pooling, reducing architectural complexity
- **Minimal Layers**: Only 2 convolutional layers and 2 fully connected layers keep the model interpretable
- **No Normalization**: No batch normalization or dropout - focuses on the core CNN learning dynamics

### 📐 Layer-by-Layer Breakdown

| Layer | Type | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|------------|
| Conv1 | Conv2d | (1, 28, 28) | (32, 28, 28) | kernel=3×3, stride=1, padding=1 |
| ReLU1 | Activation | (32, 28, 28) | (32, 28, 28) | - |
| Conv2 | Conv2d | (32, 28, 28) | (64, 14, 14) | kernel=3×3, stride=2, padding=1 |
| ReLU2 | Activation | (64, 14, 14) | (64, 14, 14) | - |
| Flatten | Reshape | (64, 14, 14) | (12544,) | - |
| FC1 | Linear | (12544,) | (128,) | - |
| ReLU3 | Activation | (128,) | (128,) | - |
| FC2 | Linear | (128,) | (10,) | Output logits |

### 🎨 Architecture Diagram

![mnist_architecture-2_page-0001](https://github.com/user-attachments/assets/6225a646-39e1-4dea-8d6e-f81b2eeffb67)

## 🔬 Environment

### Training Configuration

- **Loss Function:** MSE Loss (Mean Squared Error)
- **Optimizer:** SGD (Stochastic Gradient Descent)
- **Learning Rate:** 1×10⁻³ (0.001)
- **Batch Size:** 32
- **Epochs:** 100
- **Data Split:** 50,000 training / 10,000 testing samples

### Data Preprocessing

The data pipeline is minimal by design:
- No normalization or standardization
- No data augmentation
- Simple conversion to tensors with one-hot encoded labels
- Preserves training/test split from original MNIST dataset

### Training Hardware

- **CPU:** AMD Ryzen 5 7600
- **GPU:** AMD RX 5700XT (DirectML backend)
- **RAM:** 16GB

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.42%** |
| **Training Accuracy** | 98.46% |

### 🔍 Observations

- Minimal overfitting observed (train/test accuracy difference <0.1%)
- Architecture choices (strided conv, no pooling) work effectively for MNIST


