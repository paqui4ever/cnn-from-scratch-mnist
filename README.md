# cnn-from-scratch-mnist

## Motivation
This project implements a convolutional neural network from scratch using PyTorch to gain a deeper understanding of CNN design choices and training dynamics.

## Architecture
The model consists of two convolutional layers with ReLU activations, followed by a fully connected classifier. Spatial downsampling is achieved via strided convolution instead of pooling. The architecture balances simplicity and expressive power, achieving strong performance on MNIST while remaining easy to analyze and reproduce.

The complete architecture can be seen in the following graph:


![mnist_architecture-2_page-0001](https://github.com/user-attachments/assets/6225a646-39e1-4dea-8d6e-f81b2eeffb67)
