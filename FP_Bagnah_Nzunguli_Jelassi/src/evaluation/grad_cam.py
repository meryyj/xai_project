# src/evaluation/grad_cam.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import build_dataloaders, EMOTION_LABELS, RAFDBDataset
from src.models.factory import build_model_from_config

logger = logging.getLogger("grad_cam")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    
    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" (ICCV 2017)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The CNN model
            target_layer: The layer to compute Grad-CAM on (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[:, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        return self.generate_cam(input_tensor, target_class)


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Find the last convolutional layer in the model.
    For MainCNN, this is typically the last block in features.
    """
    # For MainCNN, features is a Sequential of ConvBlocks
    if hasattr(model, "features"):
        features = model.features
        # Find last ConvBlock
        for i in range(len(features) - 1, -1, -1):
            block = features[i]
            # Look for conv2 inside the block (the second conv layer)
            if hasattr(block, "conv2"):
                return block.conv2
    
    # Fallback: find last Conv2d in entire model
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        raise ValueError("No convolutional layer found in model")
    
    return last_conv


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, 3) in RGB, range [0, 255]
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap
    
    Returns:
        Overlayed image (H, W, 3) in RGB
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_sample_with_gradcam(
    model: nn.Module,
    grad_cam: GradCAM,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    true_label: int,
    pred_label: int,
    confidence: float,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a visualization showing original image and Grad-CAM overlay.
    
    Args:
        model: The model
        grad_cam: GradCAM instance
        image_tensor: Preprocessed image tensor
        original_image: Original image for visualization (RGB, 0-255)
        true_label: True class index
        pred_label: Predicted class index
        confidence: Prediction confidence
        device: Torch device
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Generate Grad-CAM for predicted class
    image_tensor = image_tensor.unsqueeze(0).to(device)
    heatmap = grad_cam(image_tensor, target_class=pred_label)
    
    # Overlay heatmap
    overlayed = overlay_heatmap_on_image(original_image, heatmap, alpha=0.5)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlayed)
    
    # Add prediction info
    correct = "✓" if true_label == pred_label else "✗"
    color = "green" if true_label == pred_label else "red"
    
    title = (
        f"{correct} Pred: {EMOTION_LABELS[pred_label]} ({confidence:.2%})\n"
        f"True: {EMOTION_LABELS[true_label]}"
    )
    axes[2].set_title(title, fontsize=12, fontweight="bold", color=color)
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Grad-CAM visualization to: {save_path}")
    
    return fig


def create_gradcam_gallery(
    model: nn.Module,
    dataset: RAFDBDataset,
    predictions: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    samples_per_class: int = 3,
    correct_only: bool = False,
    incorrect_only: bool = False,
) -> None:
    """
    Create a gallery of Grad-CAM visualizations for each emotion class.
    
    Args:
        model: The model
        dataset: Dataset to sample from
        predictions: Dictionary with y_true, y_pred, y_probs
        device: Torch device
        output_dir: Directory to save visualizations
        samples_per_class: Number of samples per class
        correct_only: Only show correctly predicted samples
        incorrect_only: Only show incorrectly predicted samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get last conv layer for Grad-CAM
    target_layer = get_last_conv_layer(model)
    logger.info(f"Using layer for Grad-CAM: {target_layer}")
    
    grad_cam = GradCAM(model, target_layer)
    
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    y_probs = predictions["y_probs"]
    
    # For each emotion class
    for class_idx, emotion in enumerate(EMOTION_LABELS):
        logger.info(f"Generating Grad-CAM for class: {emotion}")
        
        # Find samples of this class
        class_mask = y_true == class_idx
        
        if correct_only:
            class_mask = class_mask & (y_pred == y_true)
        elif incorrect_only:
            class_mask = class_mask & (y_pred != y_true)
        
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            logger.warning(f"No samples found for class {emotion} with given filters")
            continue
        
        # Sample randomly
        n_samples = min(samples_per_class, len(class_indices))
        sampled_indices = np.random.choice(class_indices, size=n_samples, replace=False)
        
        # Create visualizations
        for i, sample_idx in enumerate(sampled_indices):
            # Load sample
            sample = dataset.samples[sample_idx]
            image_path = sample.image_path
            
            # Load original image
            original_image = Image.open(image_path).convert("RGB")
            original_image_np = np.array(original_image)
            
            # Get preprocessed tensor
            image_tensor, label = dataset[sample_idx]
            
            # Get prediction info
            pred_idx = y_pred[sample_idx]
            true_idx = y_true[sample_idx]
            confidence = y_probs[sample_idx, pred_idx]
            
            # Generate visualization
            prefix = "correct" if pred_idx == true_idx else "incorrect"
            save_path = output_dir / f"{emotion}_{prefix}_{i+1}.png"
            
            fig = visualize_sample_with_gradcam(
                model,
                grad_cam,
                image_tensor,
                original_image_np,
                true_idx,
                pred_idx,
                confidence,
                device,
                save_path=save_path,
            )
            plt.close(fig)
        
        logger.info(f"  Generated {n_samples} visualizations for {emotion}")
    
    logger.info(f"✅ Grad-CAM gallery complete! Saved to: {output_dir}")


def analyze_attention_regions(
    heatmap: np.ndarray,
    top_k_percent: float = 0.2,
) -> Dict[str, Any]:
    """
    Analyze where the model is focusing (which regions have high activation).
    
    Args:
        heatmap: Grad-CAM heatmap (H, W)
        top_k_percent: Percentage of pixels to consider as "high attention"
    
    Returns:
        Dictionary with analysis results
    """
    # Threshold for high attention
    threshold = np.percentile(heatmap, (1 - top_k_percent) * 100)
    high_attention_mask = heatmap >= threshold
    
    # Compute center of mass of attention
    y_coords, x_coords = np.where(high_attention_mask)
    if len(y_coords) > 0:
        center_y = np.mean(y_coords) / heatmap.shape[0]
        center_x = np.mean(x_coords) / heatmap.shape[1]
    else:
        center_y = center_x = 0.5
    
    # Compute spread (standard deviation)
    if len(y_coords) > 0:
        spread_y = np.std(y_coords) / heatmap.shape[0]
        spread_x = np.std(x_coords) / heatmap.shape[1]
    else:
        spread_y = spread_x = 0.0
    
    analysis = {
        "attention_center": (center_x, center_y),
        "attention_spread": (spread_x, spread_y),
        "high_attention_ratio": float(high_attention_mask.sum() / heatmap.size),
        "max_activation": float(heatmap.max()),
        "mean_activation": float(heatmap.mean()),
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_v1.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions CSV from evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/evaluation/gradcam",
        help="Directory to save Grad-CAM visualizations",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=3,
        help="Number of samples to visualize per class",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use",
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = build_model_from_config(cfg.data).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            "Checkpoint must contain either 'model_state' or 'model_state_dict'."
        )
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load dataset (without augmentation)
    logger.info("Loading dataset...")
    loaders = build_dataloaders(cfg.data)
    dataset = loaders[args.split].dataset
    
    # Load predictions
    logger.info(f"Loading predictions from: {args.predictions_csv}")
    import pandas as pd
    pred_df = pd.read_csv(args.predictions_csv)
    
    predictions = {
        "y_true": pred_df["y_true"].values,
        "y_pred": pred_df["y_pred"].values,
        "y_probs": pred_df[[f"prob_{label}" for label in EMOTION_LABELS]].values,
    }
    
    # Create galleries
    output_dir = Path(args.output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("Generating Grad-CAM visualizations...")
    logger.info("=" * 80)
    
    # Gallery of correct predictions
    create_gradcam_gallery(
        model,
        dataset,
        predictions,
        device,
        output_dir / "correct",
        samples_per_class=args.samples_per_class,
        correct_only=True,
    )
    
    # Gallery of incorrect predictions
    create_gradcam_gallery(
        model,
        dataset,
        predictions,
        device,
        output_dir / "incorrect",
        samples_per_class=args.samples_per_class,
        incorrect_only=True,
    )
    
    logger.info(f"\n✅ All Grad-CAM visualizations complete! Saved to: {output_dir}")


if __name__ == "__main__":
    main()
