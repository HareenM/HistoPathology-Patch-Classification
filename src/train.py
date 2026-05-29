# ============================================================
# train.py
# Project: H&E Histopathology Patch Classification Baseline
#
# Purpose:
# - Train CNN/ResNet models
# - Validate after each epoch
# - Track loss and accuracy
# - Save the best model checkpoint
# ============================================================

# Path and setup libraries
from pathlib import Path
from typing import Dict, Tuple

# Machine Learning libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration and GPU enabling libraries
from src.config import DEVICE, MODEL_DIR


##### One training epoch
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = DEVICE
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns
    -------
    avg_loss:
        Average training loss for the epoch.

    accuracy:
        Training accuracy for the epoch.
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size

        predicted_labels = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix({
            "loss": loss.item()
        })

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


##### One validation epoch
def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set for one epoch.

    Returns
    -------
    avg_loss:
        Average validation loss.

    accuracy:
        Validation accuracy.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size

            predicted_labels = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix({
                "loss": loss.item()
            })

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


##### Full training loop
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    model_name: str,
    device: torch.device = DEVICE,
    save_dir: Path = MODEL_DIR
) -> Dict[str, list]:
    """
    Train a model and save the best checkpoint based on validation loss.

    Parameters
    ----------
    model:
        PyTorch model.

    train_loader:
        Training DataLoader.

    val_loader:
        Validation DataLoader.

    criterion:
        Loss function.

    optimizer:
        Optimizer.

    num_epochs:
        Number of training epochs.

    model_name:
        Name used for saving the model checkpoint.

    device:
        CPU or CUDA device.

    save_dir:
        Directory where model checkpoints are saved.

    Returns
    -------
    history:
        Dictionary containing train/validation loss and accuracy per epoch.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_model_path = save_dir / f"{model_name}_best.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    model = model.to(device)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
                "history": history
            }

            torch.save(checkpoint, best_model_path)

            print(f"Best model saved to: {best_model_path}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_model_path}")

    return history