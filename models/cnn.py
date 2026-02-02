import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple 2-layer Convolutional Neural Network for MNIST digit classification.
    
    
    Input Shape: (batch_size, 1, 28, 28)
    Output Shape: (batch_size, 10) - logits for 10 digit classes
    
    Architecture Flow:
        (1, 28, 28) → Conv2d → (32, 28, 28) → ReLU
                    → Conv2d → (64, 14, 14) → ReLU
                    → Flatten → (12544,)
                    → Linear → (128,) → ReLU
                    → Linear → (10,)
    
    Total Parameters: ~1.6M parameters
    """
    
    def __init__(self):
        """Initialize the SimpleCNN model."""
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        # Input: 1 channel (grayscale), Output: 32 channels
        # Kernel: 3x3, Stride: 1, Padding: 1
        # Maintains spatial dimensions: 28x28 → 28x28
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU()
        
        # Second convolutional layer
        # Input: 32 channels, Output: 64 channels
        # Kernel: 3x3, Stride: 2, Padding: 1
        # Downsamples spatial dimensions: 28x28 → 14x14
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.relu2 = nn.ReLU()
        
        # Flatten layer
        # Converts (64, 14, 14) → (12544,)
        self.flatten = nn.Flatten()
        
        # First fully connected layer
        # Input: 64 * 14 * 14 = 12544 features
        # Output: 128 hidden units
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        
        # Output layer
        # Input: 128 hidden units
        # Output: 10 classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        # First conv block
        x = self.conv1(x)  # (batch, 1, 28, 28) → (batch, 32, 28, 28)
        x = self.relu1(x)
        
        # Second conv block (with downsampling)
        x = self.conv2(x)  # (batch, 32, 28, 28) → (batch, 64, 14, 14)
        x = self.relu2(x)
        
        # Flatten
        x = self.flatten(x)  # (batch, 64, 14, 14) → (batch, 12544)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch, 12544) → (batch, 128)
        x = self.relu3(x)
        x = self.fc2(x)  # (batch, 128) → (batch, 10)
        
        return x
    
    def count_parameters(self):
        """
        Count the total number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device=None):
    """
    Create and initialize a SimpleCNN model.
    
    Args:
        device (torch.device, optional): Device to place model on.
                                        If None, uses CPU.
    
    Returns:
        SimpleCNN: Initialized model
    """
    model = SimpleCNN()
    
    if device is not None:
        model = model.to(device)
    
    print(f"Created SimpleCNN with {model.count_parameters():,} parameters")
    
    return model
