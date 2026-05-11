# src/evaluation/evaluate.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.datasets import build_dataloaders, EMOTION_LABELS
from src.models.factory import build_model_from_config

logger = logging.getLogger("evaluate")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a saved checkpoint and restore model weights.
    
    Returns the full checkpoint dict for inspection.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            "Checkpoint must contain either 'model_state' or 'model_state_dict'."
        )

    model.load_state_dict(state_dict)
    logger.info(f"Model weights restored from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset split.
    
    Returns:
        Dictionary containing:
        - accuracy
        - balanced_accuracy
        - macro_f1, weighted_f1
        - per_class_metrics (precision, recall, f1)
        - y_true, y_pred (if return_predictions=True)
    """
    model.eval()
    
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for i, label_name in enumerate(EMOTION_LABELS):
        per_class_metrics[label_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    
    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
    }
    
    if return_predictions:
        results["y_true"] = y_true
        results["y_pred"] = y_pred
        results["y_probs"] = y_probs
    
    return results


def print_evaluation_summary(results: Dict[str, Any], split_name: str = "Test") -> None:
    """
    Print a nicely formatted evaluation summary.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{split_name.upper()} SET EVALUATION RESULTS")
    logger.info(f"{'=' * 70}")
    
    logger.info(f"\n📊 Overall Metrics:")
    logger.info(f"  Accuracy:          {results['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1:          {results['macro_f1']:.4f}")
    logger.info(f"  Weighted F1:       {results['weighted_f1']:.4f}")
    
    logger.info(f"\n📈 Per-Class Metrics:")
    logger.info(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    logger.info("-" * 70)
    
    per_class = results["per_class_metrics"]
    for emotion in EMOTION_LABELS:
        metrics = per_class[emotion]
        logger.info(
            f"{emotion:<12} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} "
            f"{metrics['support']:<10d}"
        )
    
    logger.info(f"{'=' * 70}\n")


def save_results_to_csv(
    results: Dict[str, Any],
    output_path: str | Path,
    split_name: str = "test",
) -> None:
    """
    Save evaluation results to CSV files.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overall metrics
    overall_df = pd.DataFrame([{
        "split": split_name,
        "accuracy": results["accuracy"],
        "balanced_accuracy": results["balanced_accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
    }])
    overall_path = output_path / f"{split_name}_overall_metrics.csv"
    overall_df.to_csv(overall_path, index=False)
    logger.info(f"Saved overall metrics to: {overall_path}")
    
    # Per-class metrics
    per_class_rows = []
    for emotion, metrics in results["per_class_metrics"].items():
        row = {"emotion": emotion}
        row.update(metrics)
        per_class_rows.append(row)
    
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = output_path / f"{split_name}_per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)
    logger.info(f"Saved per-class metrics to: {per_class_path}")
    
    # Save predictions if available
    if "y_true" in results and "y_pred" in results:
        pred_df = pd.DataFrame({
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
            "correct": results["y_true"] == results["y_pred"],
        })
        
        # Add probabilities for each class
        if "y_probs" in results:
            for i, label in enumerate(EMOTION_LABELS):
                pred_df[f"prob_{label}"] = results["y_probs"][:, i]
        
        pred_path = output_path / f"{split_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to: {pred_path}")


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Generate sklearn classification report.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=EMOTION_LABELS,
        digits=4,
    )
    
    logger.info(f"\n{report}")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Saved classification report to: {output_path}")
    
    return report


def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Args:
        results_dict: Dict mapping model names to their evaluation results
        output_path: Optional path to save comparison table
    
    Returns:
        DataFrame with comparison
    """
    comparison_rows = []
    
    for model_name, results in results_dict.items():
        row = {
            "model": model_name,
            "accuracy": results["accuracy"],
            "balanced_acc": results["balanced_accuracy"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],
        }
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("macro_f1", ascending=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info("\n" + "=" * 80 + "\n")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved model comparison to: {output_path}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
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
        "--output_dir",
        type=str,
        default="experiments/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "evaluate.log"
    setup_logging(log_file)
    
    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed set to {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building model...")
    model = build_model_from_config(cfg.data).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    
    # Build dataloaders
    logger.info("Building dataloaders...")
    loaders = build_dataloaders(cfg.data)
    eval_loader = loaders[args.split]
    
    # Evaluate
    logger.info(f"Evaluating on {args.split} set...")
    results = evaluate_model(
        model,
        eval_loader,
        device,
        return_predictions=True,
    )
    
    # Print summary
    print_evaluation_summary(results, split_name=args.split)
    
    # Generate classification report
    generate_classification_report(
        results["y_true"],
        results["y_pred"],
        output_path=output_dir / f"{args.split}_classification_report.txt",
    )
    
    # Save results
    save_results_to_csv(results, output_dir, split_name=args.split)
    
    logger.info(f"✅ Evaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
