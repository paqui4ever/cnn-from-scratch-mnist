# cnn-from-scratch-mnist

## Motivation
This project implements a convolutional neural network from scratch using PyTorch to gain a deeper understanding of CNN design choices and training dynamics.

## Architecture
The model consists of two convolutional layers with ReLU activations, followed by a fully connected classifier. Spatial downsampling is achieved via strided convolution instead of pooling. The architecture balances simplicity and expressive power, achieving strong performance on MNIST while remaining easy to analyze and reproduce.

```yaml
Input
1 × 28 × 28
   │
   ▼
Conv2D (3×3, stride=1, padding=1)
32 channels
→ Output: 32 × 28 × 28
   │
   ▼
ReLU
   │
   ▼
Conv2D (3×3, stride=2, padding=1)
64 channels
→ Output: 64 × 14 × 14
   │
   ▼
ReLU
   │
   ▼
Flatten
→ 64 · 14 · 14 = 12,544
   │
   ▼
Fully Connected (12,544 → 128)
   │
   ▼
ReLU
   │
   ▼
Fully Connected (128 → 10)
   │
   ▼
Output (Logits)
10 classes

```
