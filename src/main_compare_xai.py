from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data.datasets import EMOTION_LABELS, build_dataloaders
from src.evaluation.faithfulness import compute_faithfulness, plot_faithfulness_curves
from src.evaluation.masking_analysis import analyze_masking
from src.evaluation.perturbation_xai import (
    KernelSHAPExplainer,
    LIMEExplainer,
    PerturbationExplainerConfig,
)
from src.evaluation.robustness import evaluate_robustness
from src.evaluation.xai_common import (
    GradCAMExplainer,
    collect_dataset_predictions,
    restore_model_from_checkpoint,
    sample_indices_per_class,
    save_explanation_figure,
    save_method_comparison_figure,
)
from src.models.factory import build_model_from_config
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = logging.getLogger("compare_xai")


METHOD_DISPLAY_NAMES = {
    "gradcam": "Grad-CAM",
    "lime": "LIME",
    "shap": "SHAP",
}

REGION_DISPLAY_NAMES = {
    "eyes": "Eyes",
    "mouth": "Mouth",
    "background": "Background",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XAI methods on FER predictions")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_xai.yaml",
        help="Path to config file with dataset/model/XAI settings.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/best_model150epochs.pt",
        help="Path to checkpoint of the trained FER model.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/xai",
        help="Directory to save XAI outputs.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated list of methods to run. Defaults to config xai.methods.",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="Override config for number of samples per class.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on total number of analyzed samples.",
    )
    return parser.parse_args()


def _resolve_methods(args: argparse.Namespace, cfg: dict) -> list[str]:
    if args.methods:
        return [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    return [method.strip().lower() for method in cfg.get("xai", {}).get("methods", ["gradcam", "lime", "shap"])]


def _build_explainers(model: torch.nn.Module, device: torch.device, xai_cfg: dict, methods: list[str]):
    perturbation_cfg = xai_cfg.get("perturbation_methods", {})
    perturbation = PerturbationExplainerConfig(
        rows=int(perturbation_cfg.get("rows", 5)),
        cols=int(perturbation_cfg.get("cols", 5)),
        num_samples=int(perturbation_cfg.get("num_samples", 200)),
        batch_size=int(perturbation_cfg.get("batch_size", 32)),
        baseline_strategy=str(perturbation_cfg.get("baseline_strategy", "blur")),
        blur_kernel_size=int(perturbation_cfg.get("blur_kernel_size", 11)),
        ridge_alpha=float(perturbation_cfg.get("ridge_alpha", 1.0)),
        kernel_width=float(perturbation_cfg.get("kernel_width", 5.0)),
        random_seed=int(xai_cfg.get("random_seed", 42)),
    )

    explainers = {}
    for method in methods:
        if method == "gradcam":
            explainers[method] = GradCAMExplainer(model, device)
        elif method == "lime":
            explainers[method] = LIMEExplainer(model, device, perturbation)
        elif method == "shap":
            explainers[method] = KernelSHAPExplainer(model, device, perturbation)
        else:
            raise ValueError(f"Unsupported XAI method: {method}")
    return explainers


def _write_markdown_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    masking_summary_df: pd.DataFrame,
) -> None:
    def _df_to_markdown_lines(df: pd.DataFrame) -> list[str]:
        headers = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in df.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return lines

    report_path = output_dir / "xai_summary.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# XAI Comparison Summary\n\n")
        f.write("## Method Summary\n\n")
        if summary_df.empty:
            f.write("No summary metrics were produced.\n")
        else:
            f.write("\n".join(_df_to_markdown_lines(summary_df)))
            f.write("\n\n")
        f.write("## Masking Summary\n\n")
        if masking_summary_df.empty:
            f.write("No masking metrics were produced.\n")
        else:
            f.write("\n".join(_df_to_markdown_lines(masking_summary_df)))
            f.write("\n")


def _display_method_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method).lower(), str(method))


def _display_region_name(region: str) -> str:
    return REGION_DISPLAY_NAMES.get(str(region).lower(), str(region))


def _plot_grouped_bars(
    ax,
    df: pd.DataFrame,
    metrics: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
) -> None:
    positions = list(range(len(df)))
    width = 0.8 / max(len(metrics), 1)
    colors = ["#2563EB", "#F97316", "#16A34A", "#A855F7", "#DC2626"]
    group_offset = (len(metrics) - 1) * width / 2

    for metric_idx, (metric, label) in enumerate(zip(metrics, labels)):
        offsets = [pos - group_offset + metric_idx * width for pos in positions]
        values = df[metric].astype(float).tolist()
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=label,
            color=colors[metric_idx % len(colors)],
            alpha=0.88,
        )
        for bar in bars:
            value = float(bar.get_height())
            text_offset = 3 if value >= 0 else -11
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, text_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(df["display_name"].tolist(), rotation=0)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def _save_summary_plots(
    output_dir: Path,
    summary_df: pd.DataFrame,
    masking_summary_df: pd.DataFrame,
) -> None:
    if not summary_df.empty:
        plot_df = summary_df.copy()
        plot_df["display_name"] = plot_df["method"].map(_display_method_name)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        _plot_grouped_bars(
            axes[0],
            plot_df,
            ["deletion_auc", "insertion_auc"],
            ["Deletion AUC", "Insertion AUC"],
            "Faithfulness metrics",
            "AUC",
        )
        axes[0].text(
            0.5,
            -0.20,
            "Lower deletion AUC is better; higher insertion AUC is better.",
            transform=axes[0].transAxes,
            ha="center",
            fontsize=9,
            color="#374151",
        )

        _plot_grouped_bars(
            axes[1],
            plot_df,
            [
                "robustness_pearson_mean",
                "robustness_rank_corr_mean",
                "robustness_topk_iou_mean",
            ],
            ["Pearson", "Rank corr.", "Top-k IoU"],
            "Robustness under perturbations",
            "Score",
        )
        axes[1].set_ylim(bottom=min(0.0, axes[1].get_ylim()[0]), top=1.05)

        fig.suptitle("XAI method comparison", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
        fig.savefig(output_dir / "xai_method_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not masking_summary_df.empty:
        region_order = [
            region
            for region in ["eyes", "mouth", "background"]
            if region in set(masking_summary_df["region"].astype(str))
        ]

        impact_df = (
            masking_summary_df.groupby("region", as_index=False)[
                ["confidence_drop", "prediction_changed"]
            ]
            .mean()
            .set_index("region")
            .loc[region_order]
            .reset_index()
        )
        impact_df["display_name"] = impact_df["region"].map(_display_region_name)

        mass_df = masking_summary_df.copy()
        mass_df["display_name"] = mass_df["method"].map(_display_method_name)
        mass_metrics = [f"mass_{region}" for region in region_order]
        for region, metric_name in zip(region_order, mass_metrics):
            mass_df[metric_name] = mass_df.apply(
                lambda row, region=region: row["attribution_mass"]
                if str(row["region"]) == region
                else pd.NA,
                axis=1,
            )
        mass_df = (
            mass_df.groupby(["method", "display_name"], as_index=False)[mass_metrics]
            .max()
            .sort_values("method")
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        _plot_grouped_bars(
            axes[0],
            impact_df,
            ["confidence_drop", "prediction_changed"],
            ["Confidence drop", "Prediction changed"],
            "Region masking impact",
            "Mean score",
        )
        _plot_grouped_bars(
            axes[1],
            mass_df,
            mass_metrics,
            [_display_region_name(region) for region in region_order],
            "Attribution mass by region",
            "Mean attribution mass",
        )
        axes[0].set_ylim(top=max(0.65, axes[0].get_ylim()[1]))
        axes[1].set_ylim(bottom=0.0, top=max(0.65, axes[1].get_ylim()[1]))

        fig.suptitle("Facial-region masking analysis", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
        fig.savefig(output_dir / "masking_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _sample_output_dir(output_dir: Path, row: pd.Series) -> Path:
    return output_dir / "samples" / (
        f"idx_{int(row['sample_idx']):04d}_true_{row['true_label_name']}"
        f"_pred_{row['predicted_label_name']}"
    )


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _prepare_report_figures(output_dir: Path, selected_df: pd.DataFrame) -> None:
    report_dir = output_dir / "report_figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(output_dir / "xai_method_summary.png", report_dir / "xai_method_summary.png")
    _copy_if_exists(output_dir / "masking_summary.png", report_dir / "masking_summary.png")

    confusion_candidates = [
        Path("experiments/evaluation/main_cnn_150epochs/confusion/test_confusion_normalized_true.png"),
        Path("experiments/evaluation/main_cnn_20epochs/confusion/test_confusion_normalized_true.png"),
    ]
    for candidate in confusion_candidates:
        if _copy_if_exists(candidate, report_dir / "confusion_matrix_normalized.png"):
            break

    if selected_df.empty:
        return

    for sample_idx, filename in {
        4190: "example_correct_happy_happy.png",
        449: "example_incorrect_anger_happy.png",
    }.items():
        rows = selected_df[selected_df["sample_idx"].astype(int) == sample_idx]
        if rows.empty:
            continue
        source = _sample_output_dir(output_dir, rows.iloc[0]) / "qualitative_comparison.png"
        _copy_if_exists(source, report_dir / filename)

    correct_rows = selected_df[selected_df["correct"].astype(int) == 1].sort_values(
        "predicted_confidence",
        ascending=False,
    )
    incorrect_rows = selected_df[selected_df["correct"].astype(int) == 0].sort_values(
        "predicted_confidence",
        ascending=False,
    )
    if not correct_rows.empty:
        source = _sample_output_dir(output_dir, correct_rows.iloc[0]) / "qualitative_comparison.png"
        _copy_if_exists(source, report_dir / "best_correct_example.png")
    if not incorrect_rows.empty:
        source = _sample_output_dir(output_dir, incorrect_rows.iloc[0]) / "qualitative_comparison.png"
        _copy_if_exists(source, report_dir / "best_incorrect_example.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "compare_xai.log")

    cfg = load_config(args.config)
    cfg_dict = cfg.data
    xai_cfg = cfg_dict.get("xai", {})
    methods = _resolve_methods(args, cfg_dict)

    logger.info(f"Loading config from: {args.config}")
    logger.info(f"Using methods: {methods}")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = build_model_from_config(cfg_dict).to(device)
    checkpoint = restore_model_from_checkpoint(model, args.checkpoint, device)
    model.eval()
    logger.info(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} "
        f"with keys: {sorted(checkpoint.keys())}"
    )

    loaders = build_dataloaders(cfg_dict)
    dataset = loaders[args.split].dataset
    predictions = collect_dataset_predictions(
        model,
        dataset,
        device,
        batch_size=int(xai_cfg.get("prediction_batch_size", 32)),
        num_workers=int(xai_cfg.get("prediction_num_workers", 0)),
    )

    samples_per_class = (
        args.samples_per_class
        if args.samples_per_class is not None
        else int(xai_cfg.get("samples_per_class", 2))
    )
    selected_indices = sample_indices_per_class(
        predictions["y_true"],
        samples_per_class=samples_per_class,
        max_samples=args.max_samples or xai_cfg.get("max_samples"),
        seed=int(xai_cfg.get("random_seed", 42)),
    )
    logger.info(f"Selected {len(selected_indices)} samples for XAI analysis")

    explainers = _build_explainers(model, device, xai_cfg, methods)

    faithfulness_cfg = xai_cfg.get("faithfulness", {})
    robustness_cfg = xai_cfg.get("robustness", {})
    masking_cfg = xai_cfg.get("masking", {})

    sample_rows: list[dict] = []
    robustness_rows: list[dict] = []
    masking_rows: list[dict] = []
    selected_sample_rows: list[dict] = []

    for sample_idx in selected_indices:
        image_tensor, true_label = dataset[sample_idx]
        predicted_label = int(predictions["y_pred"][sample_idx])
        predicted_confidence = float(predictions["y_probs"][sample_idx, predicted_label])

        sample_name = (
            f"idx_{sample_idx:04d}_true_{EMOTION_LABELS[int(true_label)]}"
            f"_pred_{EMOTION_LABELS[predicted_label]}"
        )
        sample_dir = output_dir / "samples" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        selected_sample_rows.append(
            {
                "sample_idx": sample_idx,
                "true_label": int(true_label),
                "true_label_name": EMOTION_LABELS[int(true_label)],
                "predicted_label": predicted_label,
                "predicted_label_name": EMOTION_LABELS[predicted_label],
                "predicted_confidence": predicted_confidence,
                "correct": int(int(true_label) == predicted_label),
            }
        )

        qualitative_results = []
        for method_name, explainer in explainers.items():
            logger.info(f"Explaining sample {sample_idx} with {method_name}")
            result = explainer.explain(image_tensor, target_class=predicted_label)
            qualitative_results.append(result)

            save_explanation_figure(
                image_tensor,
                result.heatmap,
                cfg_dict,
                sample_dir / f"{method_name}_explanation.png",
                title=(
                    f"{_display_method_name(method_name)} | "
                    f"true={EMOTION_LABELS[int(true_label)]} | "
                    f"predicted={EMOTION_LABELS[result.predicted_class]} | "
                    f"target={EMOTION_LABELS[result.target_class]} | "
                    f"p_target={result.confidence:.2%}"
                ),
            )

            faithfulness = compute_faithfulness(
                model,
                image_tensor,
                result.heatmap,
                result.target_class,
                device,
                steps=int(faithfulness_cfg.get("steps", 10)),
                baseline_strategy=str(faithfulness_cfg.get("baseline_strategy", "blur")),
                blur_kernel_size=int(faithfulness_cfg.get("blur_kernel_size", 11)),
            )
            plot_faithfulness_curves(
                faithfulness,
                sample_dir / f"{method_name}_faithfulness.png",
                title=f"{method_name} faithfulness",
            )

            robustness = evaluate_robustness(
                explainer,
                image_tensor,
                result,
                result.target_class,
                num_perturbations=int(robustness_cfg.get("num_perturbations", 5)),
                noise_std=float(robustness_cfg.get("noise_std", 0.02)),
                brightness_delta=float(robustness_cfg.get("brightness_delta", 0.03)),
                translation_px=int(robustness_cfg.get("translation_px", 2)),
                topk_ratio=float(robustness_cfg.get("topk_ratio", 0.2)),
                random_seed=int(xai_cfg.get("random_seed", 42)) + sample_idx,
            )
            for row in robustness.perturbation_rows:
                robustness_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "method": method_name,
                        **row,
                    }
                )

            region_rows = analyze_masking(
                model,
                image_tensor,
                result.heatmap,
                result.target_class,
                device,
                regions=masking_cfg.get("regions", ["eyes", "mouth", "background"]),
                baseline_strategy=str(masking_cfg.get("baseline_strategy", "blur")),
                blur_kernel_size=int(masking_cfg.get("blur_kernel_size", 11)),
            )
            for row in region_rows:
                masking_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "method": method_name,
                        "true_label_name": EMOTION_LABELS[int(true_label)],
                        "predicted_label_name": EMOTION_LABELS[predicted_label],
                        **row,
                    }
                )

            sample_rows.append(
                {
                    "sample_idx": sample_idx,
                    "method": method_name,
                    "true_label": int(true_label),
                    "true_label_name": EMOTION_LABELS[int(true_label)],
                    "predicted_label": predicted_label,
                    "predicted_label_name": EMOTION_LABELS[predicted_label],
                    "predicted_confidence": predicted_confidence,
                    "deletion_auc": faithfulness.deletion_auc,
                    "insertion_auc": faithfulness.insertion_auc,
                    "robustness_pearson_mean": robustness.pearson_mean,
                    "robustness_rank_corr_mean": robustness.rank_corr_mean,
                    "robustness_topk_iou_mean": robustness.topk_iou_mean,
                    "correct": int(int(true_label) == predicted_label),
                }
            )

        save_method_comparison_figure(
            image_tensor,
            cfg_dict,
            qualitative_results,
            sample_dir / "qualitative_comparison.png",
            true_label=int(true_label),
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
        )

    per_sample_df = pd.DataFrame(sample_rows)
    robustness_df = pd.DataFrame(robustness_rows)
    masking_df = pd.DataFrame(masking_rows)
    selected_df = pd.DataFrame(selected_sample_rows)

    selected_df.to_csv(output_dir / "selected_samples.csv", index=False)
    per_sample_df.to_csv(output_dir / "xai_per_sample_metrics.csv", index=False)
    robustness_df.to_csv(output_dir / "xai_robustness_details.csv", index=False)
    masking_df.to_csv(output_dir / "xai_masking_metrics.csv", index=False)

    summary_df = pd.DataFrame()
    if not per_sample_df.empty:
        summary_df = (
            per_sample_df.groupby("method", as_index=False)[
                [
                    "deletion_auc",
                    "insertion_auc",
                    "robustness_pearson_mean",
                    "robustness_rank_corr_mean",
                    "robustness_topk_iou_mean",
                ]
            ]
            .mean()
            .sort_values("insertion_auc", ascending=False)
        )
        summary_df.to_csv(output_dir / "xai_method_summary.csv", index=False)

    masking_summary_df = pd.DataFrame()
    if not masking_df.empty:
        masking_summary_df = (
            masking_df.groupby(["method", "region"], as_index=False)[
                ["confidence_drop", "attribution_mass", "prediction_changed"]
            ]
            .mean()
            .sort_values(["method", "region"])
        )
        masking_summary_df.to_csv(output_dir / "xai_masking_summary.csv", index=False)

    _save_summary_plots(output_dir, summary_df, masking_summary_df)
    _prepare_report_figures(output_dir, selected_df)
    _write_markdown_report(output_dir, summary_df, masking_summary_df)
    logger.info(f"XAI comparison complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
