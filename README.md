# Explainable AI for Image Classification using CNN and Grad-CAM

A beginner-friendly, clean, and modular academic semester project demonstrating **Explainable Artificial Intelligence (XAI)** in computer vision.

---

## 📌 Project Overview

Deep learning models, particularly Convolutional Neural Networks (CNNs), achieve high accuracy on visual classification tasks. However, they are traditionally treated as **"black-box"** systems because their internal decision-making process is opaque.

In high-stakes domains (e.g., medical diagnostics, autonomous driving, forensics), understanding **why** a model made a specific prediction is just as important as the prediction itself.

This project:
1. Trains a lightweight **Convolutional Neural Network (CNN)** on the standard **CIFAR-10** dataset (10 image classes).
2. Applies **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate visual heatmaps highlighting the exact pixel regions the CNN focused on.
3. Produces visual explanations displaying:
   - The **Original Image**
   - The **Predicted Class Label**
   - The **Prediction Confidence Score (%)**
   - The **Grad-CAM Attention Heatmap & Blended Overlay**

---

## 📁 Project Directory Structure

```text
explainable-ai-image-classification/
│
├── data/               # Stores downloaded CIFAR-10 dataset files
├── models/             # Stores saved model checkpoints (.pth files)
├── results/            # Stores output explanation plots and Grad-CAM figures
├── src/                # Python source modules
│   ├── __init__.py     # Package initialization
│   ├── model.py        # CNN architecture definition
│   ├── train.py        # Training pipeline and weight saving
│   ├── test.py         # Evaluation and accuracy verification on test data
│   └── gradcam.py      # Grad-CAM heatmap extraction and visualization
│
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusions for datasets, checkpoints, and caches
└── README.md           # Project documentation and presentation guide
```

---

## 📄 File Descriptions

| File / Folder | Purpose |
| :--- | :--- |
| `data/` | Holds the CIFAR-10 dataset files automatically downloaded via `torchvision`. |
| `models/` | Stores the trained weights file (e.g., `cifar10_cnn.pth`) for reuse during testing and explanation. |
| `results/` | Stores output visualizations (side-by-side plots of the original image and Grad-CAM overlay). |
| `src/model.py` | Contains the `SimpleCNN` PyTorch module with convolutional, pooling, and linear classification layers. |
| `src/train.py` | Loads CIFAR-10, defines the loss function and optimizer, executes the training loop, and saves the trained model weights. |
| `src/test.py` | Evaluates the trained model on test images, calculating accuracy and testing individual sample classifications. |
| `src/gradcam.py` | Implements the Grad-CAM algorithm: hooks the final conv layer, computes gradients for the target class, builds heatmaps, and overlays them on images. |
| `requirements.txt` | Lists standard Python packages required for the project (PyTorch, Torchvision, NumPy, Matplotlib, OpenCV, Pillow). |

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- `pip` (Python package manager)

### 2. Create and Activate Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Next Steps

Once this initial scaffold is reviewed, the project will be implemented in sequential stages:
1. **Model Definition (`src/model.py`)**: Complete CNN architecture with accessible feature maps.
2. **Training Pipeline (`src/train.py`)**: CIFAR-10 data loaders, training loop, and model checkpoint saving.
3. **Evaluation (`src/test.py`)**: Accuracy verification on test data.
4. **Grad-CAM Engine (`src/gradcam.py`)**: Gradient extraction, heatmap calculation, and side-by-side visualization.
