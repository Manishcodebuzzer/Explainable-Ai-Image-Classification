"""
gradcam.py
==========
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for SimpleCNN.

Academic Context & Theory:
--------------------------
Explainable AI (XAI) aims to make deep learning models interpretable and trustworthy.
CNNs are often criticized as "black-box" models. Grad-CAM directly answers:

    "Where exactly did the model LOOK when making this specific prediction?"

Grad-CAM Process (Step-by-Step):
---------------------------------
STEP 1 - Forward Pass:
    Pass the input image through the CNN. Extract and store the feature maps
    (activations) produced by the target convolutional layer (conv3).
    These feature maps encode high-level spatial information (e.g., where
    object parts appear in the image).

STEP 2 - Backward Pass:
    Compute the gradient of the target class score (logit) with respect to the
    feature maps of the target layer. These gradients describe how much each
    spatial location in each feature channel contributed to the final prediction.

STEP 3 - Global Average Pooling of Gradients → Importance Weights:
    Apply Global Average Pooling (GAP) over the spatial dimensions (H, W) of
    the gradient maps. This produces one scalar weight per feature channel.
    High weight = that channel was important for the predicted class.

STEP 4 - Weighted Combination of Feature Maps:
    Multiply each feature map by its corresponding importance weight and sum
    them together. This collapses 64 channels into a single 2D spatial map
    showing which regions of the image contributed most.

STEP 5 - ReLU:
    Apply ReLU to keep only positive values. Negative values would indicate
    regions that suppress the predicted class. We only care about regions
    that support the class prediction.

STEP 6 - Normalize and Overlay:
    Scale the heatmap values to [0, 1], resize it to the original 32x32 image
    size using bilinear interpolation, apply a colour map (JET), and blend
    it with the original image to produce the final explanation.

References:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization." ICCV 2017.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Ensure project root is on sys.path so imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model import SimpleCNN

# Ensure torchvision recognizes locally cached CIFAR-10 batches without re-downloading
torchvision.datasets.cifar.check_integrity = lambda fpath, md5=None: os.path.exists(fpath)

# CIFAR-10 human-readable class names in label-index order
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 test set bounds
MIN_TEST_INDEX = 0
MAX_TEST_INDEX = 9999


def get_user_test_index(min_idx: int = MIN_TEST_INDEX, max_idx: int = MAX_TEST_INDEX) -> int:
    """
    Prompts the user to enter a CIFAR-10 test image index from the terminal,
    validates the input, and exits with a clear error message if invalid.
    """
    try:
        user_input = input("Enter CIFAR-10 test image index (0-9999): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    # 1. Validate integer conversion
    try:
        idx = int(user_input)
    except ValueError:
        print(f"Error: Invalid input '{user_input}'. Please enter an integer between {min_idx} and {max_idx}.")
        sys.exit(1)

    # 2. Validate range
    if not (min_idx <= idx <= max_idx):
        print(f"Error: Index {idx} is out of range. Please enter an integer between {min_idx} and {max_idx}.")
        sys.exit(1)

    return idx


# ==============================================================================
# Grad-CAM Engine
# ==============================================================================

class GradCAM:
    """
    Grad-CAM explanation generator for a convolutional neural network.

    How it works internally:
    - Registers two PyTorch hooks on the target convolutional layer (conv3):
        1. Forward hook: captures the output feature maps during the forward pass.
        2. Backward hook: captures the gradients flowing back during backprop.
    - Combines them to produce a 2D spatial heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model       : The trained CNN model (SimpleCNN).
            target_layer: The convolutional layer whose activations and gradients
                          we want to inspect (model.conv3).
        """
        self.model = model
        self.target_layer = target_layer

        # Storage for captured feature maps and gradients
        self._feature_maps = None  # Set by forward hook
        self._gradients = None     # Set by backward hook

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers a forward hook and a backward hook on the target layer.

        Forward hook: Called AFTER the layer computes its output.
                      We save the output tensor (feature maps).

        Backward hook: Called AFTER gradients flow through the layer.
                       We save the gradient tensor w.r.t. the layer's output.
        """
        # Forward hook: save feature maps produced by conv3
        def forward_hook(module, input, output):
            # output shape: (1, 64, 4, 4) for our SimpleCNN
            self._feature_maps = output.detach()

        # Backward hook: save gradients of the class score w.r.t. conv3 outputs
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (1, 64, 4, 4)
            self._gradients = grad_output[0].detach()

        # Attach both hooks to the target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates a 2D Grad-CAM heatmap for the given input image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, 32, 32).
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            heatmap (np.ndarray): 2D array of shape (32, 32) with values in [0, 1].
            predicted_class (int): Index of the predicted class.
            confidence (float): Prediction confidence in [0, 1].
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # --- STEP 1: Forward Pass ---
        # Enable gradients so we can backpropagate to conv3's feature maps
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)  # shape: (1, 10)

        # Compute class probabilities and get the predicted class
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = predicted_class

        # --- STEP 2: Backward Pass ---
        # Zero any previous gradients
        self.model.zero_grad()

        # Extract the score for the target class
        class_score = logits[0, target_class]

        # Backpropagate to compute gradients of the target class score
        # with respect to the target layer's feature maps
        class_score.backward()

        # --- STEP 3: Global Average Pooling of Gradients → Importance Weights ---
        # self._gradients shape: (1, 64, 4, 4)
        # Average over spatial dims (H=4, W=4) → shape: (64,)
        weights = self._gradients[0].mean(dim=(1, 2))  # One weight per channel

        # --- STEP 4: Weighted Combination of Feature Maps ---
        # self._feature_maps shape: (1, 64, 4, 4)
        feature_maps = self._feature_maps[0]  # shape: (64, 4, 4)

        # Linear combination: sum over channels, each scaled by its weight
        # Result shape: (4, 4)
        heatmap = torch.zeros(feature_maps.shape[1:], device=feature_maps.device)
        for channel_idx, w in enumerate(weights):
            heatmap += w * feature_maps[channel_idx]

        # --- STEP 5: ReLU ---
        # Keep only activations that positively support the predicted class
        heatmap = F.relu(heatmap)

        # --- STEP 6: Normalize the heatmap to [0, 1] ---
        heatmap = heatmap.cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize heatmap from (4, 4) to original image size (32, 32) using bilinear interpolation
        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(heatmap_tensor, size=(32, 32), mode="bilinear", align_corners=False)
        heatmap_resized = heatmap_resized.squeeze().numpy()

        return heatmap_resized, predicted_class, confidence

    def remove_hooks(self):
        """
        Removes all registered hooks from the target layer.
        Call this when Grad-CAM is no longer needed to free resources.
        """
        for handle in self.target_layer._forward_hooks.copy():
            self.target_layer._forward_hooks.pop(handle)
        for handle in self.target_layer._backward_hooks.copy():
            self.target_layer._backward_hooks.pop(handle)


# ==============================================================================
# Visualization Helpers
# ==============================================================================

def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses CIFAR-10 normalization to restore original pixel values in [0, 1]
    for matplotlib image display.

    Formula: x_original = (x_normalized * std) + mean
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # (C, H, W) → (H, W, C) for matplotlib
    return tensor.permute(1, 2, 0).numpy()


def overlay_heatmap_on_image(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blends the Grad-CAM heatmap colour map with the original RGB image.

    Args:
        rgb_image : Original image in (H, W, 3), values in [0, 1].
        heatmap   : Normalized 2D heatmap in (H, W), values in [0, 1].
        alpha     : Transparency of the heatmap overlay (0 = invisible, 1 = full).

    Returns:
        overlay (np.ndarray): Blended image in (H, W, 3), values in [0, 1].
    """
    # Apply JET colour map: converts scalar [0,1] heatmap to RGB colours
    colormap = plt.colormaps.get_cmap("jet")
    heatmap_colored = colormap(heatmap)[:, :, :3]  # (H, W, 3), discard alpha channel

    # Blend: (1 - alpha) * original + alpha * heatmap
    overlay = (1 - alpha) * rgb_image + alpha * heatmap_colored
    return np.clip(overlay, 0.0, 1.0)


def visualize_gradcam(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    true_class: str,
    confidence: float,
    save_path: Path = None
):
    """
    Creates and saves a 3-panel Grad-CAM visualization figure:
    - Panel 1: Original Image
    - Panel 2: Grad-CAM Heatmap (coloured)
    - Panel 3: Heatmap overlay on original image

    Args:
        rgb_image       : Original unnormalized image (32, 32, 3).
        heatmap         : 2D Grad-CAM heatmap (32, 32) in [0, 1].
        predicted_class : Human-readable name of predicted class.
        true_class      : Human-readable name of true class.
        confidence      : Prediction confidence (0.0 – 1.0).
        save_path       : Full path to save the PNG file.
    """
    is_correct = (predicted_class == true_class)
    label_color = "#2e7d32" if is_correct else "#c62828"  # Green / Red
    mark = "✓ Correct" if is_correct else "✗ Incorrect"

    overlay = overlay_heatmap_on_image(rgb_image, heatmap, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("#f8f9fa")

    # Panel 1 – Original Image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2 – Grad-CAM Heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 – Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", fontsize=12, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Figure-level caption
    title = (
        f"True: {true_class}   |   Pred: {predicted_class}   |   "
        f"Confidence: {confidence * 100:.1f}%   [{mark}]"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", color=label_color, y=0.02)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(str(save_path), dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    plt.close()


# ==============================================================================
# Main Driver
# ==============================================================================

def main():
    """
    Interactive Grad-CAM pipeline:
    1. Prompts the user for a CIFAR-10 test image index (0-9999).
    2. Validates input; prints a clear error message and exits if invalid.
    3. Selects device (MPS if available, otherwise CPU).
    4. Loads trained SimpleCNN weights from models/cnn_model.pth.
    5. Loads CIFAR-10 test dataset with training normalization.
    6. Initialises Grad-CAM using conv3 as the target layer.
    7. Computes prediction and generates Grad-CAM heatmap for the selected image.
    8. Saves 3-panel visualization to results/gradcam_test_<index>.png.
    9. Prints prediction and explanation summary to the terminal.
    """
    # 1. Interactive index selection and validation
    idx = get_user_test_index(MIN_TEST_INDEX, MAX_TEST_INDEX)

    # Paths
    data_dir    = PROJECT_ROOT / "data"
    model_path  = PROJECT_ROOT / "models" / "cnn_model.pth"
    results_dir = PROJECT_ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)

    # 2. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device selected: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")

    # 3. Load trained model
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'.\n"
            "Please run 'python3 src/train.py' first."
        )

    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    print(f"Loaded trained model from: {model_path}")

    # 4. Load CIFAR-10 test dataset with training normalization
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
    print(f"CIFAR-10 test dataset loaded: {len(test_dataset)} images.")

    # 5. Initialise Grad-CAM using conv3 as the target layer
    gradcam = GradCAM(model=model, target_layer=model.conv3)

    # 6. Process the selected test image
    image, true_label = test_dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)  # shape: (1, 3, 32, 32)

    # Generate Grad-CAM heatmap
    heatmap, pred_class_idx, conf = gradcam.generate_heatmap(input_tensor)

    true_class_name = CIFAR10_CLASSES[true_label]
    pred_class_name = CIFAR10_CLASSES[pred_class_idx]
    is_correct = (pred_class_idx == true_label)

    # Convert normalized tensor back to displayable RGB image
    rgb_image = unnormalize(image)

    # 7. Save output visualization: results/gradcam_test_<index>.png
    output_filename = f"gradcam_test_{idx}.png"
    save_path = results_dir / output_filename
    visualize_gradcam(
        rgb_image=rgb_image,
        heatmap=heatmap,
        predicted_class=pred_class_name,
        true_class=true_class_name,
        confidence=conf,
        save_path=save_path
    )

    # 8. Print summary
    print("\n" + "=" * 55)
    print(f"Selected test index : {idx}")
    print(f"True class          : {true_class_name}")
    print(f"Predicted class     : {pred_class_name}")
    print(f"Confidence          : {conf * 100:.2f}%")
    print(f"Prediction correct  : {'Yes (✓)' if is_correct else 'No (✗)'}")
    print(f"Saved output path   : results/{output_filename}")
    print("=" * 55)



if __name__ == "__main__":
    main()
