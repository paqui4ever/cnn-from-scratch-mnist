import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.dataset import load_mnist_data
from models.cnn import create_model


def get_device(device_arg=None):
    """
    Detect and return the best available device.
    
    Priority:
    1. User-specified device
    2. CUDA (NVIDIA GPUs)
    3. DirectML (AMD/Intel GPUs)
    4. CPU
    
    Args:
        device_arg (str, optional): User-specified device ('cuda', 'cpu', 'directml')
    
    Returns:
        torch.device: Device to use for training
    """
    if device_arg:
        if device_arg == 'directml':
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                print("Warning: torch_directml not found, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device(device_arg)
    
    # Auto-detect best device
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device('cuda')
    
    try:
        import torch_directml
        print("Using DirectML device")
        return torch_directml.device()
    except ImportError:
        pass
    
    print("Using CPU device")
    return torch.device('cpu')


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        # Move data to device
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        float: Test accuracy percentage
    """
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_inputs = test_inputs.to(device).float()
            test_targets = test_targets.to(device).long()
            
            # Forward pass
            test_outputs = model(test_inputs)
            
            # Get predictions
            _, predicted = torch.max(test_outputs, 1)
            
            # Compare with ground truth
            correct_predictions += (predicted == torch.argmax(test_targets, dim=1)).sum().item()
            total_predictions += test_targets.size(0)
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SimpleCNN on MNIST')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'directml'],
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    print(f"Training on: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("Loading MNIST Dataset")
    print("="*50)
    train_loader, test_loader = load_mnist_data(
        batch_size=args.batch_size,
        device=device
    )
    
    # Create model
    print("\n" + "="*50)
    print("Creating Model")
    print("="*50)
    model = create_model(device=device)
    
    # Setup training
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print("="*50 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        
        # Train for one epoch
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on test set
        accuracy = evaluate(model, test_loader, device)
        
        print(f"Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
            }, best_path)
            print(f"✓ New best accuracy! Model saved to {best_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
