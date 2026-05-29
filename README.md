# Histopathology Patch Classification using Classical CNNs

## 1. Project Overview

This project builds a baseline computational pathology pipeline for classifying H&E-stained histology image patches into tissue categories using classical convolutional neural networks and transfer learning models.

The purpose of this project is to establish a strong foundation in histopathology image analysis before moving into more advanced biomedical AI workflows such as spatial transcriptomics, multimodal image-gene alignment, and histology-based representation learning.

The project uses H&E image patches organized into tissue-class folders and trains three baseline models:

1. A custom SmallCNN trained from scratch
2. ResNet-18 with transfer learning
3. ResNet-50 with transfer learning

The full workflow includes dataset loading, exploratory data analysis, preprocessing, augmentation, stratified train-validation-test splitting, model training, model evaluation, confusion matrix analysis, and Grad-CAM interpretability.

---

## 2. Motivation

Histopathology is one of the most important data modalities in biomedical research and cancer diagnostics. Tissue sections are commonly stained with hematoxylin and eosin, or H&E.

Hematoxylin highlights nuclei in blue-purple tones, while eosin highlights cytoplasm, extracellular matrix, and surrounding tissue structures in pink tones. These stain patterns allow pathologists and computational models to identify morphological differences between tissue types.

Whole-slide images are extremely large, so computational pathology pipelines often divide them into smaller image patches. Each patch captures a localized tissue region. Patch-level classification is therefore a foundational task for learning tissue morphology.

This project answers the question:

> Can CNN-based models learn discriminative morphology patterns from H&E image patches and classify them into biologically meaningful tissue categories?

---

## 3. Dataset Structure

The dataset is organized directly under `data/raw/`, where each subfolder represents one tissue class.

```text
data/
└── raw/
    ├── ADI/
    ├── BACK/
    ├── DEB/
    ├── LYM/
    ├── MUC/
    ├── MUS/
    ├── NORM/
    ├── STR/
    └── TUM/
```

Each folder contains image patches belonging to that tissue category.

---

## 4. Tissue Class Mapping

| Class Code | Biological Meaning |
|---|---|
| ADI | Adipose tissue |
| BACK | Background / empty region |
| DEB | Debris |
| LYM | Lymphocytes |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Cancer-associated stroma |
| TUM | Tumor epithelium |

This mapping is important because the model sees only integer labels during training, but the analysis requires biological interpretation of each tissue class.

---

## 5. Project Goals

The main goals of this project are:

1. Build a reproducible histopathology image classification pipeline.
2. Understand the visual behavior of H&E image patches.
3. Train a simple CNN baseline from scratch.
4. Compare the simple CNN with pretrained ResNet architectures.
5. Evaluate performance using accuracy, macro F1-score, weighted F1-score, and class-wise metrics.
6. Interpret model errors using confusion matrices.
7. Use Grad-CAM to inspect whether the model focuses on meaningful tissue regions.
8. Create a biotech-friendly project foundation for future spatial transcriptomics work.

---

## 6. Project Structure

```text
histology-patch-classification/
│
├── data/
│   └── raw/
│       ├── ADI/
│       ├── BACK/
│       ├── DEB/
│       ├── LYM/
│       ├── MUC/
│       ├── MUS/
│       ├── NORM/
│       ├── STR/
│       └── TUM/
│
├── notebooks/
│   └── 01_histology_patch_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam_utils.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install PyTorch separately using the CUDA wheel command appropriate for the machine. Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name histology-venv --display-name "Python (histology-venv)"
```

---

## 8. Main Dependencies

The project uses:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
opencv-python
tqdm
jupyter
ipykernel
torch
torchvision
torchaudio
grad-cam
```

---

## 9. Workflow Summary

The project follows this pipeline:

```text
Dataset organization
        ↓
Dataset sanity check
        ↓
Visual EDA
        ↓
Image preprocessing and augmentation
        ↓
Stratified train/validation/test split
        ↓
DataLoader construction
        ↓
SmallCNN training
        ↓
ResNet-18 training
        ↓
ResNet-50 training
        ↓
Test-set evaluation
        ↓
Confusion matrix analysis
        ↓
Grad-CAM visualization
        ↓
Final interpretation
```

---

## 10. Exploratory Data Analysis

The notebook first checks whether the dataset folders are correctly detected and whether the expected tissue classes are present.

EDA includes:

- Counting images per class
- Displaying sample patches from each tissue class
- Checking image dimensions
- Checking RGB channel consistency
- Visualizing class distribution

This step helps confirm that the classification problem is biologically meaningful and technically valid before model training.

The visual inspection is especially important in histopathology because different tissue classes may differ by nuclear density, glandular structure, fibrous texture, mucus regions, background artifacts, or tumor morphology.

---

## 11. Image Preprocessing

All images are resized to:

```text
224 × 224 pixels
```

This size is used because ResNet architectures commonly expect 224 × 224 image inputs.

The preprocessing pipeline includes:

```text
Resize
Convert to tensor
Normalize using ImageNet mean and standard deviation
```

ImageNet normalization values:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Although H&E images are different from ImageNet images, this normalization is useful because the ResNet models are initialized with ImageNet-pretrained weights.

---

## 12. Data Augmentation

Training images receive light augmentation:

```text
Random horizontal flip
Random vertical flip
Random rotation
Color jitter
```

These augmentations are appropriate for H&E patches because tissue orientation is usually not fixed, and staining intensity can vary between slides, scanners, and preparation conditions.

Validation and test images do not receive random augmentation. They only receive deterministic preprocessing so that evaluation remains stable and reproducible.

---

## 13. Train/Validation/Test Split

The dataset is split using a stratified split:

```text
Train:      70%
Validation: 15%
Test:       15%
```

Stratification preserves class proportions across all three splits.

The training set is used for model optimization.  
The validation set is used for model selection and monitoring generalization.  
The test set is held out for final evaluation.

This prevents the model from being evaluated on images that influenced training or model-selection decisions.

---

## 14. Modular Source Code Design

The project is organized into reusable Python scripts rather than placing all logic inside the notebook.

### `src/config.py`

Stores:

```text
Project paths
Image size
Batch size
Random seed
Device configuration
Class label meanings
Training hyperparameters
Output directory paths
```

This keeps configuration consistent across the project.

### `src/dataset.py`

Handles:

```text
ImageFolder dataset loading
Training transforms
Evaluation transforms
Stratified splitting
DataLoader creation
Class mapping
Split distribution tables
Image denormalization
```

This makes the dataset pipeline reusable and keeps the notebook clean.

### `src/models.py`

Defines:

```text
SmallCNN
ResNet-18
ResNet-50
Model factory function
Parameter counting helper
Model summary helper
```

This separates model architecture from training and evaluation logic.

### `src/train.py`

Handles:

```text
One training epoch
One validation epoch
Full training loop
Loss tracking
Accuracy tracking
Best checkpoint saving
Training history creation
```

The training script is model-agnostic, so the same training function can train SmallCNN, ResNet-18, and ResNet-50.

### `src/evaluate.py`

Handles:

```text
Checkpoint loading
Prediction collection
Accuracy calculation
Macro precision
Macro recall
Macro F1-score
Weighted F1-score
Classification report
Raw confusion matrix
Normalized confusion matrix
Model comparison table
CSV output saving
```

### `src/gradcam_utils.py`

Handles:

```text
Grad-CAM generation
Target layer selection
Heatmap overlay
Correct prediction examples
Incorrect prediction examples
Grad-CAM grid plotting
```

This supports visual interpretability of model predictions.

---

## 15. Model Architectures

### 15.1 SmallCNN Baseline

The SmallCNN is a lightweight convolutional neural network trained from scratch.

Its purpose is to establish a lower-bound baseline before using deeper pretrained architectures.

The architecture contains repeated blocks:

```text
Convolution
Batch normalization
ReLU activation
Max pooling
```

The final classifier contains fully connected layers with dropout.

The SmallCNN helps answer:

> Can a simple CNN learn tissue morphology patterns directly from the dataset?

---

### 15.2 ResNet-18 Transfer Learning

ResNet-18 is used as the first transfer learning baseline.

The model starts from ImageNet-pretrained weights, and the final classification head is replaced with a custom output layer for the nine histology tissue classes.

ResNet-18 is useful because it provides a stronger feature extractor while still being relatively lightweight.

It helps answer:

> Do pretrained hierarchical visual features improve histopathology patch classification?

---

### 15.3 ResNet-50 Transfer Learning

ResNet-50 is used as a deeper transfer learning baseline.

It has greater representational capacity than ResNet-18, which may help capture richer tissue morphology. However, it is more computationally expensive and can overfit if not monitored carefully.

It helps answer:

> Does a deeper pretrained CNN improve classification performance beyond ResNet-18?

---

## 16. Training Setup

The models are trained using:

```text
Loss function: Cross-entropy loss
Optimizer: Adam
Learning rate: 1e-4
Weight decay: 1e-4
Epochs: 10 by default
```

The best model checkpoint is saved based on the lowest validation loss.

Training history is tracked using:

```text
Training loss
Training accuracy
Validation loss
Validation accuracy
```

These values are plotted after training to inspect convergence and possible overfitting.

---

## 17. Evaluation Metrics

The final models are evaluated on the held-out test set using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Macro precision | Average precision across classes |
| Macro recall | Average recall across classes |
| Macro F1-score | Balanced class-wise performance |
| Weighted F1-score | F1-score weighted by class support |
| Classification report | Per-class precision, recall, F1, and support |
| Confusion matrix | Class-level error diagnosis |

Macro F1-score is emphasized because it treats each tissue class equally. This is important in histopathology because overall accuracy can hide weak performance on minority or morphologically difficult tissue classes.

---

## 18. Confusion Matrix Analysis

The confusion matrix is used to understand which tissue classes are correctly classified and which are commonly confused.

Diagonal values represent correct predictions.  
Off-diagonal values represent misclassifications.

Important biological confusions may include:

```text
STR vs MUS
NORM vs TUM
MUC vs DEB
BACK vs DEB
```

These errors can occur because some tissue classes share visual texture, staining patterns, or mixed morphology.

For example, stroma and smooth muscle may both contain elongated or fibrous structures. Normal mucosa and tumor epithelium may both show glandular patterns, making them harder to separate in isolated patches.

---

## 19. Grad-CAM Interpretability

Grad-CAM is used to visualize which regions of an image contributed most strongly to the model prediction.

The Grad-CAM workflow includes:

```text
Selecting the final convolutional layer
Computing gradients for the predicted class
Generating a heatmap
Overlaying the heatmap on the original H&E patch
Comparing correct and incorrect predictions
```

Grad-CAM helps determine whether the model focuses on biologically meaningful tissue regions or irrelevant artifacts.

For correct predictions, useful Grad-CAM behavior may include focus on:

```text
Dense cellular regions
Glandular structures
Stromal texture
Mucus regions
Smooth muscle texture
Tumor epithelium regions
```

For incorrect predictions, Grad-CAM may reveal focus on:

```text
Background artifacts
Debris
Stain-heavy regions
Mixed tissue boundaries
Morphologically ambiguous regions
```

This makes the project more relevant to computational pathology because it goes beyond accuracy and inspects model behavior visually.

---

## 20. Expected Outputs

The project generates outputs under the `results/` directory.

### Figures

```text
results/figures/
├── split_class_distribution.png
├── sample_transformed_training_batch.png
├── small_cnn_loss_curve.png
├── small_cnn_accuracy_curve.png
├── resnet18_loss_curve.png
├── resnet18_accuracy_curve.png
├── resnet50_loss_curve.png
├── resnet50_accuracy_curve.png
├── model_validation_accuracy_comparison.png
├── model_validation_loss_comparison.png
├── small_cnn_test_raw_confusion_matrix.png
├── small_cnn_test_normalized_confusion_matrix.png
├── resnet18_test_raw_confusion_matrix.png
├── resnet18_test_normalized_confusion_matrix.png
├── resnet50_test_raw_confusion_matrix.png
├── resnet50_test_normalized_confusion_matrix.png
├── final_model_macro_f1_comparison.png
├── best_model_per_class_f1_scores.png
├── best_model_gradcam_correct_predictions.png
└── best_model_gradcam_incorrect_predictions.png
```

### Metrics

```text
results/metrics/
├── small_cnn_test_summary_metrics.csv
├── small_cnn_test_classification_report.csv
├── small_cnn_test_raw_confusion_matrix.csv
├── small_cnn_test_normalized_confusion_matrix.csv
├── resnet18_test_summary_metrics.csv
├── resnet18_test_classification_report.csv
├── resnet18_test_raw_confusion_matrix.csv
├── resnet18_test_normalized_confusion_matrix.csv
├── resnet50_test_summary_metrics.csv
├── resnet50_test_classification_report.csv
├── resnet50_test_raw_confusion_matrix.csv
├── resnet50_test_normalized_confusion_matrix.csv
└── model_test_metric_comparison.csv
```

### Model Checkpoints

```text
results/models/
├── small_cnn_best.pt
├── resnet18_best.pt
└── resnet50_best.pt
```

Large model checkpoints are excluded from GitHub using `.gitignore`.

---

## 21. GitHub Version Control Notes

The following files and folders should be tracked:

```text
src/
notebooks/
README.md
requirements.txt
.gitignore
```

The following should not be pushed to GitHub:

```text
venv/
data/raw/
data/processed/
results/models/
large .pt files
__pycache__/
.ipynb_checkpoints/
```

The dataset and model checkpoints are large and should remain local unless uploaded to a storage service separately.

---

## 22. How to Run the Project

### Step 1: Clone the repository

```bash
git clone git@github.com:your-username/histology-patch-classification.git
cd histology-patch-classification
```

### Step 2: Create and activate virtual environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add dataset

Place class folders directly under:

```text
data/raw/
```

Expected structure:

```text
data/raw/ADI/
data/raw/BACK/
data/raw/DEB/
data/raw/LYM/
data/raw/MUC/
data/raw/MUS/
data/raw/NORM/
data/raw/STR/
data/raw/TUM/
```

### Step 5: Open notebook

```text
notebooks/01_histology_patch_classification.ipynb
```

Run notebook cells in order.

---

## 23. Key Results to Report

After training and evaluation, report:

```text
Best model
Test accuracy
Test macro F1-score
Test weighted F1-score
Highest-performing tissue classes
Lowest-performing tissue classes
Most common confusion pairs
Grad-CAM interpretation
```

Example result table format:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| SmallCNN | Replace with result | Replace with result | Replace with result |
| ResNet-18 | Replace with result | Replace with result | Replace with result |
| ResNet-50 | Replace with result | Replace with result | Replace with result |

The best model should be selected primarily by macro F1-score.

---

## 24. Project Interpretation

This project demonstrates that CNN-based models can learn morphology-driven features from H&E image patches. The SmallCNN provides a transparent baseline, while ResNet-18 and ResNet-50 test the benefit of transfer learning.

The confusion matrix and class-wise F1-scores provide more useful insight than accuracy alone. They reveal whether the model performs consistently across tissue classes or only performs well on visually simple categories.

Grad-CAM adds interpretability by showing whether the model focuses on tissue-relevant regions. This is important in biomedical imaging because strong numerical performance is not sufficient if the model relies on artifacts or background patterns.

---

## 25. Limitations

This project is a baseline and has several limitations:

1. The split is patch-level, not patient-level or slide-level.
2. Patch-level random splitting may overestimate generalization if related patches come from the same slide.
3. The project uses basic color normalization rather than full stain normalization.
4. External validation on another dataset is not included.
5. Grad-CAM gives qualitative interpretability but does not prove clinical reliability.
6. The models classify isolated patches rather than full-slide tissue context.

These limitations are acceptable for a beginner baseline project, but they should be acknowledged clearly in the README or report.

---

## 26. Future Work

Future extensions include:

1. Add Macenko or Reinhard stain normalization.
2. Use patient-level or slide-level splitting if metadata is available.
3. Evaluate on an external histopathology dataset.
4. Compare CNNs with Vision Transformer or Swin Transformer models.
5. Use the trained ResNet as a feature extractor for patch embeddings.
6. Cluster learned patch embeddings to study tissue morphology structure.
7. Connect H&E patch embeddings with spatial transcriptomics gene-expression profiles.
8. Build a multimodal image-gene retrieval or prediction system.

---

## 27. Resume Bullet

Built a computational pathology pipeline for H&E colorectal histology patch classification using CNN, ResNet-18, and ResNet-50 models; implemented preprocessing, augmentation, stratified splitting, macro-F1 evaluation, confusion matrix error analysis, and Grad-CAM interpretability to benchmark morphology-driven tissue classification.

---

## 28. Recruiter-Friendly Summary

This project demonstrates an end-to-end biomedical computer vision workflow for histopathology patch classification. It combines deep learning, transfer learning, medical image preprocessing, model evaluation, and interpretability. The project is designed as a foundational step toward spatial transcriptomics and multimodal computational pathology research.

---

## 29. Technical Summary

The project uses PyTorch and torchvision to classify H&E image patches into nine tissue categories. A modular codebase separates configuration, dataset handling, model definitions, training, evaluation, and Grad-CAM utilities. Models are evaluated using accuracy, macro F1-score, weighted F1-score, per-class metrics, and confusion matrices. Grad-CAM visualizations are used to inspect whether the trained models focus on tissue-relevant morphology.

---

## 30. Final Conclusion

This project establishes a complete baseline workflow for H&E histopathology patch classification. It moves from raw tissue patches to trained CNN and ResNet models, final test-set evaluation, biological error analysis, and interpretability using Grad-CAM.

The strongest value of this project is not just the final model score. The value is the complete computational pathology workflow: understanding H&E tissue classes, building a reproducible dataset pipeline, comparing baseline models, evaluating class-wise behavior, interpreting errors, and connecting model decisions back to tissue morphology.

This makes the project a strong foundation for more advanced work in spatial transcriptomics, histology-gene alignment, and multimodal biomedical AI.
