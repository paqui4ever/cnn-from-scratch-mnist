# cnn-from-scratch-mnist

## Motivation
This project implements a convolutional neural network from scratch using PyTorch to gain a deeper understanding of CNN design choices and training dynamics.

## Architecture
The model consists of two convolutional layers with ReLU activations, followed by a fully connected classifier. Spatial downsampling is achieved via strided convolution instead of pooling. The architecture balances simplicity and expressive power, achieving strong performance on MNIST while remaining easy to analyze and reproduce.

![mnist_architecture_page-0001](https://github.com/user-attachments/assets/8a6ab728-f72f-4319-9d01-c2c06456b297)

