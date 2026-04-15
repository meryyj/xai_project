# Facial Emotion Recognition (FER) with Custom CNN

This project implements a **Convolutional Neural Network (CNN) from scratch** to classify facial expressions into **7 basic emotions** (Neutral, Happy, Sad, Angry, Surprise, Fear, Disgust).

The system is designed to be trained on the **RAF-DB** dataset without using pre-trained weights (like ImageNet or VGGFace). It features a modular codebase with config-driven experiments, custom architecture design (Depthwise Separable Convolutions, Attention mechanisms), and a complete evaluation pipeline including Grad-CAM explainability.

## 📂 Project Structure

```text
├── configs/               # YAML configuration files for experiments
├── data/                  # Dataset storage
│   ├── splits/            # CSV files for train/val/test splits
│   └── processed/         # Aligned/processed face images
├── experiments/           # Training logs, checkpoints, and evaluation outputs
├── notebooks/             # Jupyter notebooks for analysis and demos
├── src/                   # Source code
│   ├── data/              # Data loading, transforms, and splitting scripts
│   ├── evaluation/        # Metrics, confusion matrix, and Grad-CAM logic
│   ├── models/            # Custom CNN architecture and attention modules
│   ├── training/          # Training loop, losses, and schedulers
│   ├── utils/             # Logging, seeding, and config parsing
│   ├── main_train.py      # Entry point for training
│   ├── main_evaluate_all.py # Entry point for full evaluation
│   └── ...
└── requirements.txt       # Python dependencies
```

## 🚀 Key Features

  * **Custom CNN Architecture**: A unified `MainCNN` class supporting:
      * Depthwise Separable Convolutions for efficiency.
      * Configurable Attention modules (SE-Block, CBAM).
      * Residual connections and flexible block depths.
  * **Config-Driven Experiments**: All hyperparameters (learning rate, batch size, model depth, augmentation) are defined in `configs/*.yaml` files.
  * **Robust Evaluation**:
      * Automated calculation of Per-class Precision/Recall/F1 and Macro-F1.
      * Confusion Matrix generation.
      * **Grad-CAM** visualization to explain model predictions.
  * **Reproducibility**: Fixed random seeds for Python, NumPy, and PyTorch.

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/simo-br91/DeepLearning_Project.git
    cd DeepLearning_Project
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have a compatible version of PyTorch installed for your system (CUDA/CPU).*

## ⚡ Usage

### 1\. Data Preparation

Before training, generate the CSV splits from your image dataset.

```bash
# Create stratified train/val/test splits
python -m src.data.create_splits --csv data/all_images.csv --out_dir data/splits
```

  * `--csv`: Path to the CSV containing all image paths and labels.
  * `--out_dir`: Directory where `train.csv`, `val.csv`, and `test.csv` will be saved.

### 2\. Training

Train the model using a configuration file. The default config is `configs/main_cnn_template.yaml`.

```bash
python -m src.main_train --config configs/main_cnn_v1.yaml
```

  * **Logs**: Saved to `experiments/logs/train.log` and TensorBoard.
  * **Checkpoints**: Best models saved to `experiments/checkpoints/`.

### 3\. Evaluation

Run the complete evaluation pipeline to generate metrics, confusion matrices, and explainability maps.

```bash
python -m src.main_evaluate_all --config configs/main_cnn_v1.yaml --checkpoint experiments/checkpoints/best_model.pt --split test
```

  * **Outputs**: Results are saved in `experiments/evaluation/` including:
      * `test_classification_report.txt`
      * `test_confusion_matrix.png`
      * `gradcam/` folder with heatmaps for correct and incorrect predictions.

### 4\. Sanity Check

To verify the data loading pipeline and model forward pass without starting a full training run:

```bash
python -m src.main_sanity
```

## ⚙️ Configuration

You can customize experiments by modifying the YAML files in `configs/`. Key parameters include:

  * **Dataset**: Image size, channels (Grayscale/RGB), normalization stats.
  * **Model**: `num_blocks`, `base_channels`, `attention_type` ("se", "cbam", "none"), `dropout`.
  * **Training**: `batch_size`, `lr`, `optimizer` (AdamW/SGD), `scheduler` (CosineAnnealing).

## 📊 Results

The project aims to outperform shallow CNN baselines and trivial majority-class classifiers. Primary success metric is **Macro-F1** score on the held-out test set.
