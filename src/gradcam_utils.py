# ============================================================
# gradcam_utils.py
# Project: H&E Histopathology Patch Classification Baseline
#
# Purpose:
# - Generate Grad-CAM visualizations for trained CNN/ResNet models
# - Overlay model attention heatmaps on H&E tissue patches
# - Support visual interpretation of correct and incorrect predictions
# ============================================================

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import cv2

from src.config import DEVICE, FIGURES_DIR, CLASS_MEANINGS
from src.dataset import denormalize_image


##### Grad-CAM core class
class GradCAM:
    """
    Basic Grad-CAM implementation for CNN-based image classifiers.

    Grad-CAM uses gradients flowing into the final convolutional layer
    to identify image regions that contributed strongly to the prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        Generate a Grad-CAM heatmap for one image tensor.

        Parameters
        ----------
        input_tensor:
            Image tensor with shape [1, 3, H, W].

        target_class:
            Class index for which Grad-CAM should be generated.
            If None, the predicted class is used.

        Returns
        -------
        cam:
            Grad-CAM heatmap as a 2D numpy array.

        predicted_class:
            Model predicted class index.

        probabilities:
            Softmax probability vector.
        """

        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(DEVICE)

        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()

        if target_class is None:
            target_class = predicted_class

        score = outputs[:, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()

        cam = torch.relu(cam)

        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, predicted_class, probabilities.detach().cpu().squeeze()

    def remove_hooks(self):
        """
        Remove registered hooks.
        """

        self.forward_hook.remove()
        self.backward_hook.remove()


##### Target layer selection
def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Select the final convolutional layer for Grad-CAM.

    Parameters
    ----------
    model:
        Trained model.

    model_name:
        Model name: small_cnn, resnet18, or resnet50.

    Returns
    -------
    target_layer:
        Final convolutional layer used for Grad-CAM.
    """

    model_name = model_name.lower()

    if model_name == "small_cnn":
        # Last Conv2d layer inside SmallCNN feature extractor
        return model.features[12]

    if model_name in ["resnet18", "resnet50"]:
        # Last residual block
        return model.layer4[-1]

    raise ValueError(
        f"Unsupported model_name for Grad-CAM: {model_name}"
    )


##### Heatmap overlay helper
def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto an RGB image.

    Parameters
    ----------
    image_rgb:
        Displayable RGB image with values between 0 and 1.

    cam:
        Grad-CAM heatmap with values between 0 and 1.

    alpha:
        Heatmap opacity.

    Returns
    -------
    overlay:
        RGB image with Grad-CAM overlay.
    """

    image_uint8 = np.uint8(255 * image_rgb)

    cam_resized = cv2.resize(
        cam,
        (image_uint8.shape[1], image_uint8.shape[0])
    )

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image_uint8,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    overlay = overlay.astype(np.float32) / 255.0

    return overlay


##### Plot one Grad-CAM example
def plot_gradcam_example(
    image_tensor: torch.Tensor,
    true_label: int,
    model: nn.Module,
    model_name: str,
    idx_to_class: Dict[int, str],
    save_path: Optional[Path] = None
):
    """
    Plot original image, Grad-CAM heatmap, and overlay for one sample.
    """

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model=model, target_layer=target_layer)

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    cam, predicted_label, probabilities = gradcam.generate(
        input_tensor=input_tensor,
        target_class=None
    )

    gradcam.remove_hooks()

    image_rgb = denormalize_image(image_tensor)
    overlay = overlay_heatmap_on_image(image_rgb, cam)

    true_class = idx_to_class[true_label]
    pred_class = idx_to_class[predicted_label]

    true_meaning = CLASS_MEANINGS.get(true_class, true_class)
    pred_meaning = CLASS_MEANINGS.get(pred_class, pred_class)

    confidence = probabilities[predicted_label].item()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original\nTrue: {true_class}")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nPred: {pred_class} ({confidence:.2f})"
    )
    axes[2].axis("off")

    status = "Correct" if true_label == predicted_label else "Incorrect"

    plt.suptitle(
        f"{model_name} Grad-CAM | {status}\n"
        f"True: {true_class} ({true_meaning}) | "
        f"Predicted: {pred_class} ({pred_meaning})",
        fontsize=12
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


##### Find correct and incorrect examples
def collect_prediction_examples(
    model: nn.Module,
    data_loader,
    max_examples: int = 5,
    device: torch.device = DEVICE
):
    """
    Collect correct and incorrect prediction examples from a DataLoader.

    Returns
    -------
    correct_examples:
        List of tuples: (image_tensor, true_label, predicted_label)

    incorrect_examples:
        List of tuples: (image_tensor, true_label, predicted_label)
    """

    model.eval()

    correct_examples = []
    incorrect_examples = []

    with torch.no_grad():
        for images, labels in data_loader:
            images_device = images.to(device)
            labels_device = labels.to(device)

            outputs = model(images_device)
            predictions = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                image_tensor = images[i].cpu()
                true_label = labels[i].item()
                predicted_label = predictions[i].item()

                if true_label == predicted_label:
                    if len(correct_examples) < max_examples:
                        correct_examples.append(
                            (image_tensor, true_label, predicted_label)
                        )
                else:
                    if len(incorrect_examples) < max_examples:
                        incorrect_examples.append(
                            (image_tensor, true_label, predicted_label)
                        )

                if (
                    len(correct_examples) >= max_examples
                    and len(incorrect_examples) >= max_examples
                ):
                    return correct_examples, incorrect_examples

    return correct_examples, incorrect_examples


# ============================================================
# 6. Plot multiple Grad-CAM examples
# ============================================================

def plot_gradcam_grid(
    examples: List[Tuple[torch.Tensor, int, int]],
    model: nn.Module,
    model_name: str,
    idx_to_class: Dict[int, str],
    title: str,
    save_path: Optional[Path] = None
):
    """
    Plot a grid of Grad-CAM overlays for multiple examples.
    """

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model=model, target_layer=target_layer)

    n = len(examples)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (image_tensor, true_label, _) in enumerate(examples):
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        cam, predicted_label, probabilities = gradcam.generate(
            input_tensor=input_tensor,
            target_class=None
        )

        image_rgb = denormalize_image(image_tensor)
        overlay = overlay_heatmap_on_image(image_rgb, cam)

        true_class = idx_to_class[true_label]
        pred_class = idx_to_class[predicted_label]
        confidence = probabilities[predicted_label].item()

        axes[row_idx, 0].imshow(image_rgb)
        axes[row_idx, 0].set_title(f"Original\nTrue: {true_class}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(cam, cmap="jet")
        axes[row_idx, 1].set_title("Grad-CAM")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(
            f"Overlay\nPred: {pred_class} ({confidence:.2f})"
        )
        axes[row_idx, 2].axis("off")

    gradcam.remove_hooks()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()