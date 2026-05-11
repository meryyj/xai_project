# Facial Emotion Recognition – Project Charter

## Objective
Train a **from-scratch Convolutional Neural Network (CNN)** to classify facial expressions into **7 basic emotions**:
- Neutral, Happy, Sad, Angry, Surprise, Fear, Disgust

The model must be trained **without any pre-trained weights** (no ImageNet, no VGGFace, etc.) and evaluated on a fixed test set from **RAF-DB** (and optionally other datasets for generalization).

## Constraints
- No transfer learning / pre-trained backbones.
- Single-GPU friendly (reasonable memory + training time per experiment).
- Fixed input size: **128×128** images (aligned face crops, grayscale or RGB – to be decided and kept consistent).
- Only images of faces (after detection + alignment).

## Primary Success Criteria
- **Primary metric**: Macro-F1 on the held-out test set.
- **Secondary metrics**: 
  - Overall accuracy
  - Balanced accuracy
  - Weighted-F1
  - Per-class precision/recall/F1
- **Success goal**:
  - Significantly outperform:
    - A trivial baseline (always predicting the majority class).
    - A shallow CNN baseline.
    - (Optionally) a classical ML baseline (HOG/LBP + SVM).

## Scope
Included:
- Data preprocessing: face detection, landmark-based alignment, cropping, resizing, normalization.
- Data splitting: stratified train/validation/test with saved splits.
- From-scratch CNN design, training, and tuning.
- Handling class imbalance (class weights, augmentation, possibly focal loss).
- Evaluation, error analysis, confusion matrices.
- Explainability via Grad-CAM.
- Written report with figures and tables.

Out of scope (for now / future work):
- Video-based FER (temporal modeling).
- Multi-modal models (audio + video).
- Large-scale deployment and productization.

## Datasets
- **Primary**: RAF-DB (basic expression subset, aligned faces).
- **Optional secondary**: FER2013 / FER+ / CK+ for robustness and cross-dataset tests.

## Reproducibility
- All experiments must be:
  - Config-driven (`configs/*.yaml`).
  - Logged (TensorBoard / CSV).
  - Run with fixed RNG seeds (Python, NumPy, framework).
- Final report must state:
  - Dataset versions
  - Exact splits used
  - Final config for the best model
