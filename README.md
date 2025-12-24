# cnn-from-scratch-mnist

## Motivation
This project implements a convolutional neural network from scratch using PyTorch to gain a deeper understanding of CNN design choices and training dynamics.

## Architecture
The model consists of two convolutional layers with ReLU activations, followed by a fully connected classifier. Spatial downsampling is achieved via strided convolution instead of pooling. The architecture balances simplicity and expressive power, achieving strong performance on MNIST while remaining easy to analyze and reproduce.

[mnist_architecture.pdf](https://github.com/user-attachments/files/24333196/mnist_architecture.pdf)
