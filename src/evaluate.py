# ============================================================
# evaluate.py
# Project: H&E Histopathology Patch Classification Baseline
#
# Purpose:
# - Load trained checkpoints
# - Evaluate models on validation/test sets
# - Compute accuracy, macro F1, weighted F1
# - Generate classification reports
# - Generate and save confusion matrices
# - Save metrics as CSV files
# ============================================================

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.config import DEVICE, MODEL_DIR, METRICS_DIR, FIGURES_DIR


##### Load checkpoint
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device = DEVICE
) -> Dict:
    """
    Load a saved model checkpoint into a model architecture.

    Parameters
    ----------
    model:
        Model architecture that matches the saved checkpoint.

    checkpoint_path:
        Path to the saved .pt checkpoint.

    device:
        CPU or CUDA device.

    Returns
    -------
    checkpoint:
        Full checkpoint dictionary.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return checkpoint


##### Collect predictions
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference and collect true labels, predicted labels,
    and prediction probabilities.

    Parameters
    ----------
    model:
        Trained PyTorch model.

    data_loader:
        Validation or test DataLoader.

    device:
        CPU or CUDA device.

    Returns
    -------
    y_true:
        Ground-truth class indices.

    y_pred:
        Predicted class indices.

    y_prob:
        Softmax probabilities for all classes.
    """

    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_prob.extend(probabilities.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    return y_true, y_pred, y_prob


##### Compute summary metrics
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str
) -> Dict:
    """
    Compute core classification metrics.

    Metrics include:
    - accuracy
    - macro precision
    - macro recall
    - macro F1
    - weighted F1

    Macro F1 is especially important for multi-class histology because
    it gives equal weight to each tissue class.
    """

    metrics = {
        "model_name": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    return metrics


##### Classification report
def create_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Create a class-wise classification report as a DataFrame.

    Parameters
    ----------
    y_true:
        Ground-truth labels.

    y_pred:
        Predicted labels.

    class_names:
        Class names ordered by class index.

    Returns
    -------
    report_df:
        DataFrame containing precision, recall, F1-score, and support.
    """

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.reset_index().rename(columns={"index": "class"})

    return report_df


##### Confusion matrix
def create_confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = False
) -> pd.DataFrame:
    """
    Create a confusion matrix DataFrame.

    Parameters
    ----------
    normalize:
        If True, normalize by true-label row counts.
    """

    if normalize:
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            normalize="true"
        )
    else:
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names)))
        )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    return cm_df


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    model_name: str,
    split_name: str,
    normalize: bool = False,
    save_dir: Path = FIGURES_DIR
):
    """
    Plot and save a confusion matrix heatmap.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))

    fmt = ".2f" if normalize else "d"

    sns.heatmap(
        cm_df,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        square=True
    )

    matrix_type = "Normalized" if normalize else "Raw"

    plt.title(f"{matrix_type} Confusion Matrix: {model_name} on {split_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = (
        f"{model_name}_{split_name}_"
        f"{'normalized' if normalize else 'raw'}_confusion_matrix.png"
    )

    plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


##### Save evaluation outputs
def save_evaluation_outputs(
    metrics: Dict,
    report_df: pd.DataFrame,
    cm_raw_df: pd.DataFrame,
    cm_norm_df: pd.DataFrame,
    model_name: str,
    split_name: str,
    save_dir: Path = METRICS_DIR
):
    """
    Save metrics, classification report, and confusion matrices as CSV files.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        save_dir / f"{model_name}_{split_name}_summary_metrics.csv",
        index=False
    )

    report_df.to_csv(
        save_dir / f"{model_name}_{split_name}_classification_report.csv",
        index=False
    )

    cm_raw_df.to_csv(
        save_dir / f"{model_name}_{split_name}_raw_confusion_matrix.csv"
    )

    cm_norm_df.to_csv(
        save_dir / f"{model_name}_{split_name}_normalized_confusion_matrix.csv"
    )


##### Full evaluation function
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    model_name: str,
    split_name: str = "test",
    device: torch.device = DEVICE,
    save_outputs: bool = True,
    plot_matrices: bool = True
) -> Dict:
    """
    Evaluate a trained model on a validation or test split.

    Returns
    -------
    results:
        Dictionary containing metrics, report DataFrame, confusion matrices,
        predictions, and probabilities.
    """

    y_true, y_pred, y_prob = get_predictions(
        model=model,
        data_loader=data_loader,
        device=device
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        split_name=split_name
    )

    report_df = create_classification_report_df(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names
    )

    cm_raw_df = create_confusion_matrix_df(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=False
    )

    cm_norm_df = create_confusion_matrix_df(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=True
    )

    if save_outputs:
        save_evaluation_outputs(
            metrics=metrics,
            report_df=report_df,
            cm_raw_df=cm_raw_df,
            cm_norm_df=cm_norm_df,
            model_name=model_name,
            split_name=split_name
        )

    if plot_matrices:
        plot_confusion_matrix(
            cm_df=cm_raw_df,
            model_name=model_name,
            split_name=split_name,
            normalize=False
        )

        plot_confusion_matrix(
            cm_df=cm_norm_df,
            model_name=model_name,
            split_name=split_name,
            normalize=True
        )

    results = {
        "metrics": metrics,
        "classification_report": report_df,
        "raw_confusion_matrix": cm_raw_df,
        "normalized_confusion_matrix": cm_norm_df,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    return results


##### Compare multiple model results
def compare_model_metrics(
    results_dict: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Combine summary metrics from multiple evaluated models.

    Parameters
    ----------
    results_dict:
        Dictionary where keys are model names and values are outputs from
        evaluate_model().

    save_path:
        Optional path to save the comparison table.

    Returns
    -------
    comparison_df:
        DataFrame comparing model-level metrics.
    """

    rows = []

    for model_name, result in results_dict.items():
        row = result["metrics"].copy()
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    metric_cols = [
        "model_name",
        "split",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1"
    ]

    comparison_df = comparison_df[metric_cols]
    comparison_df = comparison_df.sort_values(
        by="macro_f1",
        ascending=False
    ).reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(save_path, index=False)

    return comparison_df