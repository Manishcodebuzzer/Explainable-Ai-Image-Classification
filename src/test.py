"""
test.py
=======
Evaluation pipeline for CIFAR-10 image classification using the trained SimpleCNN.

Academic Context:
- Evaluates the trained model on unseen test images (10,000 images) to measure
  generalization performance without data leakage.
- Computes overall test accuracy across all 10 CIFAR-10 classes.
- Computes prediction confidence scores using the Softmax function over raw logits.
- Saves a sample predictions visualization figure to `results/` for presentation.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model import SimpleCNN

# Ensure torchvision recognizes locally cached CIFAR-10 batches without re-downloading
torchvision.datasets.cifar.check_integrity = lambda fpath, md5=None: os.path.exists(fpath)

# CIFAR-10 human-readable class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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


def get_test_dataset(data_dir: Path):
    """
    Loads CIFAR-10 test dataset using the exact same normalization used in training.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    return test_dataset


def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses CIFAR-10 normalization to restore original pixel values in [0, 1]
    for matplotlib image display.

    Formula: x = (normalized * std) + mean
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    # Invert normalization
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Transpose from (C, H, W) to (H, W, C) for matplotlib
    return tensor.permute(1, 2, 0).numpy()


def calculate_test_accuracy(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Computes overall classification accuracy on the complete test dataset.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def visualize_sample_predictions(
    model: nn.Module,
    test_dataset,
    device: torch.device,
    num_samples: int = 5,
    save_path: Path = None
):
    """
    Selects test images and generates a clean visualization displaying:
    - Original image
    - Actual class
    - Predicted class
    - Confidence score (%)
    """
    model.eval()
    if save_path:
        os.makedirs(save_path.parent, exist_ok=True)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3.8))
    if num_samples == 1:
        axes = [axes]

    print("\nSample Test Predictions:")
    print("-" * 65)

    with torch.no_grad():
        for i in range(num_samples):
            image, label = test_dataset[i]
            input_tensor = image.unsqueeze(0).to(device)  # Add batch dim: (1, 3, 32, 32)

            # Forward pass: get raw logits
            logits = model(input_tensor)

            # Compute probabilities using Softmax
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

            confidence_pct = confidence.item() * 100.0
            predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
            actual_class = CIFAR10_CLASSES[label]

            is_correct = (predicted_class == actual_class)
            mark = "✓" if is_correct else "✗"
            print(
                f"Sample {i+1:02d}: True = {actual_class:<11} | "
                f"Pred = {predicted_class:<11} | "
                f"Conf = {confidence_pct:6.2f}% [{mark}]"
            )

            # Convert normalized tensor back to displayable RGB image
            rgb_image = unnormalize(image)

            # Display image in subplot
            axes[i].imshow(rgb_image)
            axes[i].axis("off")

            # Green title for correct, red for incorrect
            title_color = "#2e7d32" if is_correct else "#c62828"
            axes[i].set_title(
                f"Actual: {actual_class}\nPred: {predicted_class}\nConf: {confidence_pct:.1f}%",
                fontsize=11,
                fontweight="bold",
                color=title_color,
                pad=8
            )

    plt.suptitle("CIFAR-10 Test Sample Predictions (SimpleCNN)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"\nVisualization figure saved to: {save_path}")

    plt.close()


def main():
    """
    Main evaluation pipeline:
    1. Selects device (MPS or CPU).
    2. Loads CIFAR-10 test data.
    3. Loads saved model weights from models/cnn_model.pth.
    4. Computes test accuracy across all 10,000 images.
    5. Visualizes sample predictions and saves figure to results/.
    """
    # Paths
    data_dir = PROJECT_ROOT / "data"
    model_path = PROJECT_ROOT / "models" / "cnn_model.pth"
    results_dir = PROJECT_ROOT / "results"
    visualization_path = results_dir / "sample_predictions.png"

    # 1. Device selection
    device = get_device()

    # 2. Check model weights file
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python3 src/train.py' first."
        )

    # 3. Load test dataset and dataloader
    print("Loading CIFAR-10 test dataset...")
    test_dataset = get_test_dataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Test dataset loaded: {len(test_dataset)} total images.")

    # 4. Initialize and load model
    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    print(f"Loaded trained model weights from: {model_path}")

    # 5. Evaluate accuracy across all 10,000 test images
    print("\nEvaluating model on test dataset...")
    test_accuracy = calculate_test_accuracy(model, test_loader, device)
    print("=" * 65)
    print(f"Overall Test Accuracy: {test_accuracy:.2f}% (on 10,000 test images)")
    print("=" * 65)

    # 6. Visualize sample predictions and save plot
    visualize_sample_predictions(
        model=model,
        test_dataset=test_dataset,
        device=device,
        num_samples=5,
        save_path=visualization_path
    )


if __name__ == "__main__":
    main()
