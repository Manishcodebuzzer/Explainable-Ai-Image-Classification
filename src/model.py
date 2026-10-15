"""
model.py
========
Defines a beginner-friendly Convolutional Neural Network (CNN) architecture
for CIFAR-10 classification.

Academic Context:
- A CNN extracts spatial visual features (edges, textures, object parts)
  using convolutional and pooling layers.
- For Grad-CAM, the final convolutional layer (`conv3`) must be easily accessible
  so that forward activations and backward gradients can be captured.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple 3-block CNN for classifying CIFAR-10 images (3x32x32).

    Architecture:
    - Block 1: Conv2d(3 -> 16) -> ReLU -> MaxPool2d(2x2) -> Output: (16, 16, 16)
    - Block 2: Conv2d(16 -> 32) -> ReLU -> MaxPool2d(2x2) -> Output: (32, 8, 8)
    - Block 3: Conv2d(32 -> 64) -> ReLU -> MaxPool2d(2x2) -> Output: (64, 4, 4)
    - Classifier: Flatten -> Linear(64 * 4 * 4 -> 10) -> Output: (10 raw logits)
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # --- Convolutional Block 1 ---
        # Input: (3, 32, 32) -> Extracts low-level features (edges, color gradients)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples to (16, 16, 16)

        # --- Convolutional Block 2 ---
        # Input: (16, 16, 16) -> Extracts mid-level features (textures, simple patterns)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples to (32, 8, 8)

        # --- Convolutional Block 3 (Target layer for Grad-CAM) ---
        # Input: (32, 8, 8) -> Extracts high-level semantic features (object parts)
        # Named `conv3` so it can be directly accessed during Grad-CAM analysis
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples to (64, 4, 4)

        # --- Classification Head ---
        # Linear layer mapping flattened features (64 * 4 * 4 = 1024) to class logits
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleCNN.

        Args:
            x (torch.Tensor): Input batch of images of shape (B, 3, 32, 32)

        Returns:
            torch.Tensor: Raw class logits of shape (B, 10)
        """
        # Block 1: Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2: Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3: Conv -> ReLU -> MaxPool
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten feature maps into a 1D vector per image: (B, 64 * 4 * 4) -> (B, 1024)
        x = torch.flatten(x, start_dim=1)

        # Compute raw logits for the 10 classes
        x = self.fc(x)

        return x


if __name__ == "__main__":
    # 1. Instantiate the model
    model = SimpleCNN()
    print("SimpleCNN model created successfully.")

    # 2. Create a random dummy input tensor with shape (1, 3, 32, 32)
    dummy_input = torch.randn(1, 3, 32, 32)

    # 3. Pass the input through the model
    output = model(dummy_input)

    # 4. Print the output shape
    print("Output shape:", output.shape)
