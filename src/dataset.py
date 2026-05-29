# ============================================================
# dataset.py
# Project: H&E Histopathology Patch Classification Baseline
#
# Purpose:
# - Load H&E patch images from class folders
# - Define preprocessing and augmentation transforms
# - Create stratified train/validation/test splits
# - Build PyTorch DataLoaders
# - Return class mapping information for downstream modeling
# ============================================================

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    CLASS_MEANINGS,
)


# 1. Transform definitions
def get_train_transform() -> transforms.Compose:
    """
    Create the training transform pipeline.

    Training transforms include light augmentation because histology patches
    can appear in different orientations and with minor stain variation.
    """

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Create the validation/test transform pipeline.

    Evaluation transforms are deterministic. No random augmentation is used
    because validation and test metrics should reflect stable model behavior.
    """

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# Dataset loading
def load_full_dataset(raw_dir: Path = RAW_DIR) -> ImageFolder:
    """
    Load the full histology patch dataset using ImageFolder.

    Expected folder structure:

        data/raw/
        ├── ADI/
        ├── BACK/
        ├── DEB/
        ├── LYM/
        ├── MUC/
        ├── MUS/
        ├── NORM/
        ├── STR/
        └── TUM/

    Parameters
    ----------
    raw_dir:
        Path to the raw dataset folder.

    Returns
    -------
    ImageFolder
        Full dataset without transforms.
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    dataset = ImageFolder(
        root=raw_dir,
        transform=None
    )

    if len(dataset) == 0:
        raise ValueError(f"No images found in raw data directory: {raw_dir}")

    return dataset


# Class mapping utilities
def get_class_mapping(dataset: ImageFolder) -> Tuple[Dict[int, str], pd.DataFrame]:
    """
    Create class index mapping and biological meaning table.

    Parameters
    ----------
    dataset:
        ImageFolder dataset.

    Returns
    -------
    idx_to_class:
        Dictionary mapping integer class index to class code.

    class_mapping_df:
        DataFrame containing class index, class code, and biological meaning.
    """

    idx_to_class = {
        idx: class_name
        for class_name, idx in dataset.class_to_idx.items()
    }

    class_mapping_df = pd.DataFrame({
        "class_index": list(idx_to_class.keys()),
        "class_code": list(idx_to_class.values())
    })

    class_mapping_df["biological_meaning"] = (
        class_mapping_df["class_code"].map(CLASS_MEANINGS)
    )

    class_mapping_df = (
        class_mapping_df
        .sort_values("class_index")
        .reset_index(drop=True)
    )

    return idx_to_class, class_mapping_df


# Stratified train/validation/test split
def create_stratified_splits(
    dataset: ImageFolder,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train, validation, and test splits.

    The default split is:

        train = 70%
        validation = 15%
        test = 15%

    Parameters
    ----------
    dataset:
        Full ImageFolder dataset.

    test_size:
        Fraction of full dataset reserved for testing.

    val_size:
        Fraction of full dataset reserved for validation.

    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    train_indices, val_indices, test_indices:
        Arrays of dataset indices for each split.

    train_targets, val_targets, test_targets:
        Arrays of labels for each split.
    """

    targets = np.array([label for _, label in dataset.samples])
    indices = np.arange(len(dataset))

    temp_size = val_size + test_size

    train_indices, temp_indices, train_targets, temp_targets = train_test_split(
        indices,
        targets,
        test_size=temp_size,
        random_state=random_seed,
        stratify=targets
    )

    relative_test_size = test_size / temp_size

    val_indices, test_indices, val_targets, test_targets = train_test_split(
        temp_indices,
        temp_targets,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=temp_targets
    )

    return (
        train_indices,
        val_indices,
        test_indices,
        train_targets,
        val_targets,
        test_targets
    )


# Split distribution helper
def get_split_distribution(
    split_targets: np.ndarray,
    split_name: str,
    idx_to_class: Dict[int, str]
) -> pd.DataFrame:
    """
    Create a class distribution table for one dataset split.

    Parameters
    ----------
    split_targets:
        Class labels for a split.

    split_name:
        Name of the split, such as 'train', 'validation', or 'test'.

    idx_to_class:
        Dictionary mapping class index to class code.

    Returns
    -------
    DataFrame
        Class distribution table.
    """

    counts = pd.Series(split_targets).value_counts().sort_index()

    distribution_df = pd.DataFrame({
        "class_index": counts.index,
        "num_images": counts.values
    })

    distribution_df["class_code"] = distribution_df["class_index"].map(idx_to_class)
    distribution_df["biological_meaning"] = (
        distribution_df["class_code"].map(CLASS_MEANINGS)
    )
    distribution_df["split"] = split_name

    return distribution_df


def get_all_split_distributions(
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    idx_to_class: Dict[int, str]
) -> pd.DataFrame:
    """
    Create one combined class distribution table for all splits.
    """

    train_dist = get_split_distribution(train_targets, "train", idx_to_class)
    val_dist = get_split_distribution(val_targets, "validation", idx_to_class)
    test_dist = get_split_distribution(test_targets, "test", idx_to_class)

    split_distribution_df = pd.concat(
        [train_dist, val_dist, test_dist],
        ignore_index=True
    )

    return split_distribution_df


# Main DataLoader function
def create_dataloaders(
    raw_dir: Path = RAW_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    random_seed: int = RANDOM_SEED,
    val_size: float = 0.15,
    test_size: float = 0.15
):
    """
    Create train, validation, and test DataLoaders.

    This is the main function that should be called from the notebook.

    Parameters
    ----------
    raw_dir:
        Path to raw class-folder dataset.

    batch_size:
        Number of images per batch.

    num_workers:
        Number of DataLoader workers. Use 0 on Windows notebooks.

    random_seed:
        Random seed for reproducible splits.

    val_size:
        Fraction of full dataset used for validation.

    test_size:
        Fraction of full dataset used for testing.

    Returns
    -------
    train_loader:
        DataLoader for training data.

    val_loader:
        DataLoader for validation data.

    test_loader:
        DataLoader for test data.

    metadata:
        Dictionary containing class names, mappings, split indices,
        split targets, and distribution tables.
    """

    # Load dataset once without transforms to read labels and class mapping
    full_dataset = load_full_dataset(raw_dir)

    idx_to_class, class_mapping_df = get_class_mapping(full_dataset)

    (
        train_indices,
        val_indices,
        test_indices,
        train_targets,
        val_targets,
        test_targets
    ) = create_stratified_splits(
        dataset=full_dataset,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed
    )

    # Create separate base datasets so train receives augmentation
    # while validation/test receive deterministic transforms.
    train_base_dataset = ImageFolder(
        root=raw_dir,
        transform=get_train_transform()
    )

    eval_base_dataset = ImageFolder(
        root=raw_dir,
        transform=get_eval_transform()
    )

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(eval_base_dataset, val_indices)
    test_dataset = Subset(eval_base_dataset, test_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    split_distribution_df = get_all_split_distributions(
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        idx_to_class=idx_to_class
    )

    metadata = {
        "class_names": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "idx_to_class": idx_to_class,
        "class_mapping_df": class_mapping_df,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_targets": train_targets,
        "val_targets": val_targets,
        "test_targets": test_targets,
        "split_distribution_df": split_distribution_df,
        "num_classes": len(full_dataset.classes),
        "num_total_images": len(full_dataset),
        "num_train_images": len(train_indices),
        "num_val_images": len(val_indices),
        "num_test_images": len(test_indices),
    }

    return train_loader, val_loader, test_loader, metadata


# Batch visualization helper
def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor image back to displayable RGB format.

    Parameters
    ----------
    tensor_img:
        Tensor image with shape [3, H, W].

    Returns
    -------
    np.ndarray
        RGB image with shape [H, W, 3] and values between 0 and 1.
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = tensor_img.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    return img.permute(1, 2, 0).numpy()