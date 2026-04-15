# src/main_evaluate_all.py

"""
Complete evaluation pipeline:
1. Load model and evaluate on test set
2. Generate confusion matrices
3. Create Grad-CAM visualizations
4. Produce comprehensive report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from src.utils.logging import setup_logging

logger = logging.getLogger("evaluate_all")


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and log the result."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Command failed with return code {result.returncode}")
        sys.exit(1)
    else:
        logger.info(f"✅ {description} completed successfully\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline"
    )
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
        help="Base directory for evaluation outputs",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=3,
        help="Number of Grad-CAM samples per class",
    )
    parser.add_argument(
        "--skip_gradcam",
        action="store_true",
        help="Skip Grad-CAM generation (faster)",
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "evaluate_all.log"
    setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("COMPLETE EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Split: {args.split}")
    logger.info("="*80 + "\n")
    
    # Check that checkpoint exists
    if not Path(args.checkpoint).is_file():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Step 1: Model Evaluation
    # -------------------------------------------------------------------------
    predictions_csv = output_dir / f"{args.split}_predictions.csv"
    
    eval_cmd = [
        sys.executable, "-m", "src.evaluation.evaluate",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--output_dir", str(output_dir),
        "--split", args.split,
    ]
    
    run_command(eval_cmd, "Model Evaluation")
    
    # -------------------------------------------------------------------------
    # Step 2: Confusion Matrix Analysis
    # -------------------------------------------------------------------------
    confusion_dir = output_dir / "confusion"
    
    confusion_cmd = [
        sys.executable, "-m", "src.evaluation.confusion",
        "--predictions_csv", str(predictions_csv),
        "--output_dir", str(confusion_dir),
        "--prefix", f"{args.split}_confusion",
        "--top_k", "10",
    ]
    
    run_command(confusion_cmd, "Confusion Matrix Analysis")
    
    # -------------------------------------------------------------------------
    # Step 3: Grad-CAM Visualizations (optional)
    # -------------------------------------------------------------------------
    if not args.skip_gradcam:
        gradcam_dir = output_dir / "gradcam"
        
        gradcam_cmd = [
            sys.executable, "-m", "src.evaluation.grad_cam",
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--predictions_csv", str(predictions_csv),
            "--output_dir", str(gradcam_dir),
            "--samples_per_class", str(args.samples_per_class),
            "--split", args.split,
        ]
        
        run_command(gradcam_cmd, "Grad-CAM Visualization")
    else:
        logger.info("⏭️  Skipping Grad-CAM generation (--skip_gradcam flag set)\n")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("🎉 EVALUATION PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("\nGenerated outputs:")
    logger.info(f"  📊 Metrics: {output_dir}/{args.split}_overall_metrics.csv")
    logger.info(f"  📈 Per-class: {output_dir}/{args.split}_per_class_metrics.csv")
    logger.info(f"  🔢 Predictions: {predictions_csv}")
    logger.info(f"  📉 Confusion matrices: {confusion_dir}/")
    
    if not args.skip_gradcam:
        logger.info(f"  🔥 Grad-CAM visualizations: {gradcam_dir}/")
    

if __name__ == "__main__":
    main()