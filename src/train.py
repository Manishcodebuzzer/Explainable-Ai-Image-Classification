"""
train.py
========
Training pipeline for CIFAR-10 image classification using SimpleCNN.

Academic Context:
- Dataset: CIFAR-10 contains 60,000 32x32 color images across 10 classes
  (50,000 for training, 10,000 for testing).
- Loss Function: CrossEntropyLoss combines Softmax and Negative Log-Likelihood Loss,
  measuring the error between predicted class logits and true class labels.
- Optimizer: Adam (Adaptive Moment Estimation) updates the model parameters
  using adaptive learning rates.
- Model Checkpoint: The learned weights are saved to `models/cnn_model.pth` for
  evaluation and Grad-CAM visual explanation.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Ensure project root directory is in sys.path so modules can be imported smoothly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model import SimpleCNN

# Ensure torchvision recognizes locally cached CIFAR-10 batches without re-downloading
torchvision.datasets.cifar.check_integrity = lambda fpath, md5=None: os.path.exists(fpath)


def get_device() -> torch.device:
    """
    Selects the best available computation device:
    - Uses Apple Silicon GPU (MPS) if available on macOS.
    - Falls back to CPU otherwise.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device selected: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")
    return device


def get_data_loaders(data_dir: Path, batch_size: int = 64):
    """
    Downloads and prepares CIFAR-10 training and test DataLoaders.

    Transformations:
    1. ToTensor(): Converts PIL images to PyTorch tensors in range [0.0, 1.0].
    2. Normalize(): Standardizes RGB channels using CIFAR-10 mean and std values.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # CIFAR-10 training dataset (50,000 images)
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    # CIFAR-10 test dataset (10,000 images)
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    # Wrap datasets with DataLoaders for batch processing
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluates model performance on the test dataset without tracking gradients.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train():
    """
    Main training function:
    1. Sets up paths and device (Apple MPS or CPU).
    2. Loads CIFAR-10 dataset into DataLoaders.
    3. Initializes SimpleCNN, CrossEntropyLoss, and Adam optimizer.
    4. Trains for 5 epochs and prints training/test metrics after each epoch.
    5. Saves the trained model state dictionary to models/cnn_model.pth.
    """
    # 1. Setup paths
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    model_save_path = models_dir / "cnn_model.pth"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Select device (MPS or CPU)
    device = get_device()

    # 3. Load CIFAR-10 data
    print("\nLoading CIFAR-10 dataset...")
    batch_size = 64
    train_loader, test_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size)
    print(f"Dataset ready. Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # 4. Initialize model, loss function, and optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training loop
    num_epochs = 5
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 75)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients from previous step
            optimizer.zero_grad()

            # Forward pass: predict class logits
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()

            # Optimizer step: update weights
            optimizer.step()

            # Accumulate metrics
            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train

        # Evaluate on test set after each epoch
        epoch_test_loss, epoch_test_acc = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%"
        )

    print("-" * 75)

    # 6. Save model state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining completed successfully!")
    print(f"Model state dictionary saved to: {model_save_path}")


if __name__ == "__main__":
    train()
