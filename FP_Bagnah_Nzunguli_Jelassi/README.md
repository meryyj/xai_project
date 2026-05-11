# Explainable Facial Emotion Recognition

Final Project Report - XAI course

Team members:

- Mathys Bagnah
- Sydney Nzunguli
- Meriem Jelassi

This project studies explainable artificial intelligence methods for facial emotion recognition. The goal is to train a convolutional neural network on facial expression images and compare several explanation methods in order to understand which image regions influence the model predictions.

The project focuses on seven facial emotion classes: anger, disgust, fear, happy, neutral, sad, and surprise.

## Project Overview

The repository contains a complete pipeline for:

- preparing train, validation, and test splits;
- training CNN baselines and a configurable main CNN model;
- evaluating classification performance;
- generating visual explanations with Grad-CAM and perturbation-based XAI methods;
- comparing explanation quality using faithfulness, masking, and robustness analyses.

The main contribution is the comparison of generic XAI explanations with facial-region-based perturbation strategies. This makes the explanations easier to interpret in the context of facial emotion recognition, because the analysis can be related to meaningful regions such as eyes, mouth, nose, and eyebrows.

## Repository Structure

```text
.
+-- configs/                 # YAML configuration files
+-- data/                    # Dataset metadata and splits, if available locally
+-- notebooks/               # Exploration notebooks, not required for execution
+-- presentation/            # Presentation material
+-- report/                  # Final report source files
+-- src/                     # Main Python source code
|   +-- data/                # Dataset loading, transforms, split creation
|   +-- evaluation/          # Metrics, Grad-CAM, XAI evaluation tools
|   +-- models/              # CNN architectures and model factory
|   +-- training/            # Training loop, losses, schedulers
|   +-- utils/               # Configuration, logging, reproducibility helpers
+-- requirements.txt         # Python dependencies
+-- PROJECT_PLAN.md          # Project planning notes
```

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not installed correctly for your machine, install the CPU or CUDA version recommended on the official PyTorch website.

## Data

The code expects image metadata and split files under `data/`:

```text
data/
├── all_images.csv
├── processed/
│   └── processed/
│       ├── anger/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

The CSV files should contain the image paths and labels used by the dataset loader.
In the provided submission folder, the YAML configs use `dataset.root: data/processed/processed`.

To regenerate the splits from a global CSV file:

```bash
python -m src.data.create_splits --csv data/all_images.csv --out_dir data/splits
```

## Training

Train the main CNN model with:

```bash
python -m src.main_train --config configs/main_cnn_v1.yaml
```

The configuration file defines the dataset paths, image size, model architecture, optimizer, scheduler, batch size, number of epochs, and logging paths.

## Evaluation

After training, evaluate the generated checkpoint with:

```bash
python -m src.main_evaluate_all --config configs/main_cnn_v1.yaml --checkpoint experiments/checkpoints/best_model.pt --split test
```

This computes classification metrics and can generate confusion matrices and Grad-CAM visualizations.
If no checkpoint is provided in the submission folder, this command should be run after training has produced a model checkpoint under `experiments/checkpoints/`.

## XAI Comparison

Run the explainability comparison pipeline with:

```bash
python -m src.main_compare_xai --config configs/main_cnn_xai.yaml
```

The XAI pipeline compares explanation methods such as Grad-CAM, LIME-style perturbations, SHAP-style perturbations, DeepSHAP, and facial-region perturbations when enabled in the configuration.

## Reproducibility

The code uses YAML configuration files and fixed random seeds to make experiments reproducible. Main parameters are stored in `configs/`, while the implementation is organized into modular files under `src/`.

## Main Dependencies

- Python
- PyTorch and torchvision
- NumPy
- pandas
- scikit-learn
- scikit-image
- SHAP
- OpenCV
- matplotlib
- PyYAML
- MediaPipe, for facial-region-based perturbation experiments

## Submission Notes

For a lightweight code submission, include only:

- `src/`
- `configs/`
- `requirements.txt`
- `README.md`
- `PROJECT_PLAN.md`, if useful

Do not include virtual environments, Python cache files, experiment logs, checkpoints, generated figures, or large datasets unless explicitly requested.
