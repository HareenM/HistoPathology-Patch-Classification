# ============================================================
# models.py
# Project: H&E Histopathology Patch Classification Baseline
#
# Purpose:
# - Define CNN and ResNet models for H&E patch classification
# - Provide a clean model factory function
# - Keep model architecture separate from notebook logic
# ============================================================

##### Machine Learning and Deep Learning libraries
import torch
import torch.nn as nn
from torchvision import models

##### Device configuration
from src.config import DEVICE


##### Small CNN baseline
class SmallCNN(nn.Module):
    """
    A simple convolutional neural network for H&E patch classification.

    This model is intentionally lightweight. It acts as a from-scratch
    baseline before using transfer learning models such as ResNet-18.

    Input:
        RGB image tensor with shape [batch_size, 3, 224, 224]

    Output:
        Class logits with shape [batch_size, num_classes]
    """

    # Initialization method to define the layers of the SmallCNN model
    def __init__(self, num_classes: int):
        super(SmallCNN, self).__init__()

        self.features = nn.Sequential( # Sequential container to hold the convolutional feature extraction layers
            # Block 1: 224 -> 112
            nn.Conv2d( #Convolutional layer to extract features from the input image
                in_channels=3, # Input has 3 channels (RGB)
                out_channels=32, # We will learn 32 feature maps in the first convolutional layer
                kernel_size=3, # Using a 3x3 kernel to capture local patterns in the image
                stride=1, # Stride of 1 means the kernel will move one pixel ata time allowing for detailed feature extraction
                padding=1 # Keeps the spatial dimensions the same after convolution (224x224 -> 224x224)
            ),
            nn.BatchNorm2d(32), # Batch normalization to stabilize training and improve convergence, normalizes the output of the layer
            nn.ReLU(inplace=True), # ReLU activation to introduce non-linearity, allowing the model to learn complex patterns
            nn.MaxPool2d(kernel_size=2, stride=2), # Max pooling to downsample the feature maps by a factor of 2, reducing the spatial dimensions from 224x224 to 112x112

            # Block 2: 112 -> 56
            nn.Conv2d( # Second convolutional layer to learn more complex features from the output of the first block
                in_channels=32, # Input has 32 channels from the previous layer
                out_channels=64, # We will learn 64 feature maps in the second convolutional layer
                kernel_size=3, # Using a 3x3 kernel again
                stride=1, # kernel will move one pixel at a time
                padding=1 # keeps the spatial dimensions the same
            ),
            nn.BatchNorm2d(64), # batch normalization to stabilize training
            nn.ReLU(inplace=True), # ReLU activation to introduce non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2), # Max pooling to downsample the feature maps by a factor of 2

            # Block 3: 56 -> 28
            nn.Conv2d( # 3rd convolutional layer
                in_channels=64, # Input has 64 channels
                out_channels=128, # Output will have 128 feature maps
                kernel_size=3, # Using 3x3 kernel again
                stride=1, # kernel will move one pixel at a time
                padding=1 # keeps the spatial dimensions the same
            ),
            nn.BatchNorm2d(128), # batch normalization to stabilize training
            nn.ReLU(inplace=True), # ReLU activation for non-leniarity
            nn.MaxPool2d(kernel_size=2, stride=2), # Max pooling to downsample the feature maps by a factor of 2

            # Block 4: 28 -> 14
            nn.Conv2d( # 4th convolutional layer
                in_channels=128, # Input has 128 channels
                out_channels=256, # Output has 256 feature maps
                kernel_size=3, # using 3x3 kernel again
                stride=1, # kernel will move one pixel at a time
                padding=1 # keeps the spatial dimensions the same
            ),
            nn.BatchNorm2d(256), # batch normalization to stabilize training
            nn.ReLU(inplace=True), # ReLU activation for non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling to downsample the feature maps by a factor of 2
        )

        # The classifier head takes the output of the convolutional feature extractor and produces class logits
        self.classifier = nn.Sequential( # Sequential container to hold the fully connected layers of the classifier head
            nn.Flatten(), # Flattening the output of the convolutional layers to a 1D vector so that it can be fed to the fully connected layers

            nn.Linear(256 * 14 * 14, 512), # takes flattened features and maps them to a 512-D hidden layer
            nn.ReLU(inplace=True), # ReLU activation for non-linearity
            nn.Dropout(p=0.5), # Dropout to prevent overfitting by randomly setting 50% of activations to zero during training

            nn.Linear(512, 128), # maps the 512-D hidden layer to a smaller 128-D hidden layer
            nn.ReLU(inplace=True), # ReLU activation for non-linearity
            nn.Dropout(p=0.3), # Dropout of 30% to prevent overfitting

            nn.Linear(128, num_classes) # Final linear layer that maps the 128-D hidden layer to the number of output classes
        )

    # Forward method to define how the input data flows through the model layers during the forward pass
    def forward(self, x):
        x = self.features(x) # Pass the input through the convolutional feature extractor
        x = self.classifier(x) # Pass the extracted features through the classifier head for final class logits
        return x


##### ResNet-18 transfer learning model
def build_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Build a ResNet-18 model for histology patch classification.

    Parameters
    ----------
    num_classes:
        Number of tissue classes.

    pretrained:
        If True, loads ImageNet-pretrained weights.

    freeze_backbone:
        If True, freezes convolutional layers and only trains the final head.

    Returns
    -------
    model:
        ResNet-18 model with modified final classification layer.
    """

    # If pretrained is True, we load the default pretrained weights, otherwise we set weights to None
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # Loading the ResNet-18 with either pretrained or no weights
    model = models.resnet18(weights=weights)

    # If true, we freeze the backbone by setting requires_grad to False for all parameters, which means only the
        # final classification head will be trained, and the convolutional layers will not be updated during training
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # The original ResNet-18 has a final fully connected layer (model.fc).
        # We needed to replace it with a new one that matches the number of classes in our project
    in_features = model.fc.in_features

    # We now create a new fully connected layer that takes the same number of input features as the original ResNet-18's final layer,
        # but then maps it to 256-D hidden layer, applys ReLU and 30% dropout, and maps to the number of output classes for our project
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


#### ResNet-50 transfer learning model
def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Build a ResNet-50 model for histology patch classification.

    ResNet-50 is deeper than ResNet-18 and may perform better, but it is
    computationally heavier. It should be used after the simpler baselines.
    """

    # If pretrained is True, we load the default weights, else we set the weights to None
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    # Loading the ResNet-50 with either pretrained or no weights
    model = models.resnet50(weights=weights)

    # If true, we make only the final classification head trainable by freezing all the convolutional layers
        # making only the final fullly connected layer trainable
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # The original ResNet-50 has a final fully connected layer (model.fc)
    in_features = model.fc.in_features

    # We replace model.fc with a new fully connected layer that takes the same number of input features but maps it to a 512-D hidden
        # layer, applies ReLU and a 40% dropout, and then maps to the number of output classes for our project
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )

    return model


##### Model factory
def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: torch.device = DEVICE
):
    """
    Create and return a model by name.

    Parameters
    ----------
    model_name:
        Name of the model. Options:
        - "small_cnn"
        - "resnet18"
        - "resnet50"

    num_classes:
        Number of output classes.

    pretrained:
        Whether to use pretrained ImageNet weights for ResNet models.

    freeze_backbone:
        Whether to freeze pretrained convolutional layers.

    device:
        CPU or CUDA device.

    Returns
    -------
    model:
        PyTorch model moved to the selected device.
    """

    # Normalizing the model name to lowercase to allow for case-insensitive matching
    model_name = model_name.lower()

    # if small_cnn is selected, we create an instance of the SmallCNN class with the specified number of output classes
    if model_name == "small_cnn":
        model = SmallCNN(num_classes=num_classes)

    # if resnet18 is selected, we create an instance of the ResNet-18 model
    elif model_name == "resnet18":
        model = build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    # if resnet50 is selected, we create an instance of the ResNet-50 model
    elif model_name == "resnet50":
        model = build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    # if the model name doesn't match any of the 3 supported options, we raise a ValueError
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            "Choose from: 'small_cnn', 'resnet18', or 'resnet50'."
        )

    model = model.to(device)

    return model


##### Trainable parameter counter
def count_parameters(model: nn.Module):
    """
    Count total and trainable parameters in a PyTorch model.

    Returns
    -------
    dict:
        Dictionary with total, trainable, and frozen parameter counts.
    """

    # We calculate the total number of parameters by summing the number of elements in each parameter tensor
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params
    }


##### Model summary print helper
def print_model_summary(model: nn.Module, model_name: str):
    """
    Print a compact summary of the model parameter count.
    """

    # We use the count_parameters function to get the total, trainable, and frozen parameter counts for the model
    param_counts = count_parameters(model)

    print(f"Model: {model_name}")
    print(f"Total parameters: {param_counts['total_params']:,}")
    print(f"Trainable parameters: {param_counts['trainable_params']:,}")
    print(f"Frozen parameters: {param_counts['frozen_params']:,}")