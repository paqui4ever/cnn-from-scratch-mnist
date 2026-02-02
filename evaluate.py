import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from data.dataset import load_mnist_data
from models.cnn import SimpleCNN


def get_device(device_arg=None):
    """
    Detect and return the best available device.
    
    Args:
        device_arg (str, optional): User-specified device
    
    Returns:
        torch.device: Device to use
    """
    if device_arg:
        if device_arg == 'directml':
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                return torch.device('cpu')
        else:
            return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    
    return torch.device('cpu')


def evaluate_model(model, test_loader, device):
    """
    Evaluate model and collect predictions.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to use
    
    Returns:
        tuple: (accuracy, all_predictions, all_labels, all_images)
    """
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    all_predictions = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).long()
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(targets, dim=1)
            
            # Track accuracy
            correct_predictions += (predicted == true_labels).sum().item()
            total_predictions += targets.size(0)
            
            # Store for visualization
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            all_images.extend(inputs.cpu().numpy())
    
    accuracy = (correct_predictions / total_predictions) * 100
    
    return accuracy, np.array(all_predictions), np.array(all_labels), np.array(all_images)


def plot_predictions(images, predictions, labels, num_samples=16, save_path=None):
    """
    Plot sample predictions.
    
    Args:
        images: Array of images
        predictions: Predicted labels
        labels: True labels
        num_samples: Number of samples to plot
        save_path: Path to save figure (if None, displays instead)
    """
    # Randomly sample images
    indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            img_idx = indices[idx]
            img = images[img_idx][0]  # Remove channel dimension
            pred = predictions[img_idx]
            true = labels[img_idx]
            
            ax.imshow(img, cmap='gray')
            color = 'green' if pred == true else 'red'
            ax.set_title(f'Pred: {pred} | True: {true}', color=color, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sample predictions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(labels, predictions, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        save_path: Path to save figure
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_misclassified(images, predictions, labels, num_samples=16, save_path=None):
    """
    Plot misclassified examples.
    
    Args:
        images: Array of images
        predictions: Predicted labels
        labels: True labels
        num_samples: Number of misclassified samples to plot
        save_path: Path to save figure
    """
    # Find misclassified indices
    misclassified_idx = np.where(predictions != labels)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified examples found!")
        return
    
    # Sample misclassified examples
    sample_size = min(num_samples, len(misclassified_idx))
    sample_idx = np.random.choice(misclassified_idx, size=sample_size, replace=False)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(sample_size)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(sample_idx):
            img_idx = sample_idx[idx]
            img = images[img_idx][0]
            pred = predictions[img_idx]
            true = labels[img_idx]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Pred: {pred} | True: {true}', color='red', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Misclassified Examples (Total: {len(misclassified_idx)})', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Misclassified examples saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate SimpleCNN on MNIST')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'directml'],
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples to visualize (default: 16)')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots instead of saving')
    
    args = parser.parse_args()
    
    # Create output directory
    if not args.show_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    print(f"Evaluating on: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("Loading MNIST Dataset")
    print("="*50)
    _, test_loader = load_mnist_data(batch_size=args.batch_size, device=device)
    
    # Load model
    print("\n" + "="*50)
    print("Loading Model")
    print("="*50)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return
    
    model = SimpleCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'accuracy' in checkpoint:
        print(f"Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluating Model")
    print("="*50)
    
    accuracy, predictions, labels, images = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {np.sum(predictions == labels)}/{len(labels)}")
    print(f"Misclassified: {np.sum(predictions != labels)}")
    
    # Generate visualizations
    print("\n" + "="*50)
    print("Generating Visualizations")
    print("="*50)
    
    save_predictions = None if args.show_plots else os.path.join(args.output_dir, 'sample_predictions.png')
    save_confusion = None if args.show_plots else os.path.join(args.output_dir, 'confusion_matrix.png')
    save_misclassified = None if args.show_plots else os.path.join(args.output_dir, 'misclassified.png')
    
    plot_predictions(images, predictions, labels, 
                    num_samples=args.num_samples, 
                    save_path=save_predictions)
    
    plot_confusion_matrix(labels, predictions, save_path=save_confusion)
    
    plot_misclassified(images, predictions, labels,
                      num_samples=args.num_samples,
                      save_path=save_misclassified)
    
    if not args.show_plots:
        print(f"\n✓ All results saved to {args.output_dir}/")
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)


if __name__ == '__main__':
    main()
