# ============================================================
# config.py
# Project: H&E Histopathology Patch Classification Baseline
#
# Purpose:
# - Store project paths
# - Store model/data configuration
# - Store class label meanings
# - Store device configuration
# ============================================================

from pathlib import Path
import torch


# Project paths
# config.py is inside src/
# parents[1] moves one level up to the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"


# Create required output directories
for path in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    MODEL_DIR,
    METRICS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


# Dataset configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Use 0 for Windows + VS Code notebooks
RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Class label meanings
CLASS_MEANINGS = {
    "ADI": "Adipose tissue",
    "BACK": "Background / empty region",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "STR": "Cancer-associated stroma",
    "TUM": "Tumor epithelium",
}


# ImageNet normalization values
# These are used because ResNet models are commonly pretrained on ImageNet.
# Even for the custom CNN baseline, using the same normalization keeps the
# preprocessing pipeline consistent across models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Training configuration
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 3


# Model configuration
MODEL_NAME = "small_cnn"

SUPPORTED_MODELS = [
    "small_cnn",
    "resnet18",
    "resnet50",
]


# Sanity-check print helper
def print_config():
    """
    Print key project configuration values.
    Useful for verifying that paths and device settings are correct.
    """

    print("Project root:", PROJECT_ROOT)
    print("Raw data directory:", RAW_DIR)
    print("Results directory:", RESULTS_DIR)
    print("Figures directory:", FIGURES_DIR)
    print("Model directory:", MODEL_DIR)
    print("Metrics directory:", METRICS_DIR)

    print("\nImage size:", IMG_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Number of workers:", NUM_WORKERS)
    print("Random seed:", RANDOM_SEED)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nClass labels:")
    for class_code, meaning in CLASS_MEANINGS.items():
        print(f"{class_code}: {meaning}")