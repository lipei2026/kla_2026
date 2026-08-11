import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "train" / "scores"
FIGURES_DIR = PROJECT_ROOT / "figures"
ESM2_DIR = PROJECT_ROOT / "train" / "trained_models" / "ESM2"
ESM2_ABLATION_DIR = PROJECT_ROOT / "train" / "trained_models" / "ESM2_ablation"
THRESHOLD = 0.5

LOSS_EXPERIMENTS = {
    "E0 CLS": (ESM2_DIR, "fold_*_5_grouped_lr5e-05_bs32_epochs40"),
    "E1 Center-K": (
        ESM2_ABLATION_DIR,
        "fold_*_5_grouped_lr5e-05_bs32_epochs40",
    ),
    "E2 Masked mean": (
        ESM2_ABLATION_DIR,
        "E2_masked_mean_fold_*_5_grouped_lr5e-05_bs32_epochs40",
    ),
    "E3 CLS + Center-K": (
        ESM2_ABLATION_DIR,
        "E3_cls_center_k_fold_*_5_grouped_lr5e-05_bs32_epochs40",
    ),
    "E4 Center-K + mean": (
        ESM2_ABLATION_DIR,
        "E4_center_k_mean_fold_*_5_grouped_lr5e-05_bs32_epochs40",
    ),
}

MODEL_FILES = {
    "ACF": "ACF_y_label&score5.csv",
    "AAINDEX": "AAINDEX_y_label&score5.csv",
    "CKSAAP": "CKSAAP_y_label&score5.csv",
    "OBC": "OBC_y_label&score5.csv",
    "PSEAAC": "PSEAAC_y_label&score5.csv",
    "LSTM": "LSTM_y_label&score5.csv",
    "ESM2": "ESM2_y_label&score5.csv",
    "MetaModel": "meta_model_y_label&score5.csv",
}


def load_score_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    columns = {column.lower(): column for column in df.columns}
    if "label" not in columns or "score" not in columns:
        raise ValueError(f"{file_path.name} must contain 'label' and 'score' columns.")

    result = pd.DataFrame(
        {
            "label": df[columns["label"]].astype(int),
            "score": df[columns["score"]].astype(float),
        }
    )
    return result


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "mcc": matthews_corrcoef(labels, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
    }


def plot_roc_curves(score_map: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")

    for model_name in metrics_df["model"]:
        df = score_map[model_name]
        fpr, tpr, _ = roc_curve(df["label"], df["score"])
        auc_value = metrics_df.loc[metrics_df["model"] == model_name, "roc_auc"].iloc[0]
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_pr_curves(score_map: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 7), dpi=300)

    for model_name in metrics_df["model"]:
        df = score_map[model_name]
        precision, recall, _ = precision_recall_curve(df["label"], df["score"])
        pr_auc = metrics_df.loc[metrics_df["model"] == model_name, "pr_auc"].iloc[0]
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df[["model", "roc_auc", "pr_auc", "mcc"]].copy()
    plot_df = plot_df.sort_values("roc_auc", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.24

    plt.figure(figsize=(11, 6), dpi=300)
    plt.bar(x - width, plot_df["roc_auc"], width=width, label="ROC-AUC")
    plt.bar(x, plot_df["pr_auc"], width=width, label="PR-AUC")
    plt.bar(x + width, plot_df["mcc"], width=width, label="MCC")

    plt.xticks(x, plot_df["model"], rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def fold_from_trainer_state_path(trainer_state_path: Path) -> int:
    run_name = trainer_state_path.parts[-3]
    match = re.search(r"(?:^|_)fold_(\d+)_5(?:_|$)", run_name)
    if match is None:
        raise ValueError(f"Cannot determine fold from run directory: {run_name}")
    return int(match.group(1))


def discover_latest_trainer_states(base_dir: Path, run_pattern: str) -> list[Path]:
    """Return the latest retained Trainer state for every fold/run directory."""
    trainer_states = {}
    for run_dir in base_dir.glob(run_pattern):
        for state_path in run_dir.glob("checkpoint-*/trainer_state.json"):
            checkpoint_step = int(state_path.parent.name.split("-")[-1])
            previous = trainer_states.get(run_dir)
            if previous is None or checkpoint_step > previous[0]:
                trainer_states[run_dir] = (checkpoint_step, state_path)
    return [
        item[1]
        for item in sorted(
            trainer_states.values(),
            key=lambda pair: fold_from_trainer_state_path(pair[1]),
        )
    ]


def extract_loss_history(trainer_state_path: Path, experiment: str = "E0 CLS") -> pd.DataFrame:
    with trainer_state_path.open("r", encoding="utf-8") as file:
        trainer_state = json.load(file)

    fold = fold_from_trainer_state_path(trainer_state_path)
    rows = []
    for log_item in trainer_state.get("log_history", []):
        epoch = log_item.get("epoch")
        step = log_item.get("step")
        if "loss" in log_item:
            rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "epoch": epoch,
                    "step": step,
                    "metric": "train_loss",
                    "value": float(log_item["loss"]),
                }
            )
        if "eval_loss" in log_item:
            rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "epoch": epoch,
                    "step": step,
                    "metric": "eval_loss",
                    "value": float(log_item["eval_loss"]),
                }
            )
    return pd.DataFrame(rows)


def plot_esm2_loss_curves(output_png: Path, output_csv: Path) -> None:
    trainer_state_paths = discover_latest_trainer_states(
        ESM2_DIR,
        "fold_*_5_grouped_lr5e-05_bs32_epochs40",
    )
    if not trainer_state_paths:
        print(f"Skip ESM2 loss plot: no trainer_state.json found in {ESM2_DIR}")
        return

    history_frames = [extract_loss_history(path) for path in trainer_state_paths]
    history_df = pd.concat(history_frames, ignore_index=True)
    history_df.to_csv(output_csv, index=False)

    _, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=False)
    for fold in sorted(history_df["fold"].unique()):
        fold_df = history_df[history_df["fold"] == fold]
        train_df = fold_df[fold_df["metric"] == "train_loss"]
        eval_df = fold_df[fold_df["metric"] == "eval_loss"]

        axes[0].plot(train_df["epoch"], train_df["value"], linewidth=1.8, label=f"Fold {fold}")
        axes[1].plot(eval_df["epoch"], eval_df["value"], linewidth=1.8, label=f"Fold {fold}")

    axes[0].set_title("ESM2 Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("ESM2 Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()


def load_ablation_loss_history() -> pd.DataFrame:
    history_frames = []
    for experiment, (base_dir, run_pattern) in LOSS_EXPERIMENTS.items():
        trainer_state_paths = discover_latest_trainer_states(base_dir, run_pattern)
        if len(trainer_state_paths) != 5:
            raise FileNotFoundError(
                f"Expected 5 trainer states for {experiment}, found {len(trainer_state_paths)}"
            )
        history_frames.extend(
            extract_loss_history(path, experiment) for path in trainer_state_paths
        )
    return pd.concat(history_frames, ignore_index=True)


def plot_esm2_ablation_losses(
    output_folds_png: Path,
    output_mean_png: Path,
    output_csv: Path,
) -> None:
    """Plot full five-fold histories and mean +/- SD for E0/E1/E2/E4."""
    history_df = load_ablation_loss_history()
    history_df.to_csv(output_csv, index=False)

    experiments = list(LOSS_EXPERIMENTS)
    _, axes = plt.subplots(len(experiments), 2, figsize=(13, 15), dpi=300)
    for row, experiment in enumerate(experiments):
        experiment_df = history_df[history_df["experiment"] == experiment]
        for fold in range(1, 6):
            fold_df = experiment_df[experiment_df["fold"] == fold]
            for column, metric in enumerate(["train_loss", "eval_loss"]):
                metric_df = fold_df[fold_df["metric"] == metric]
                axes[row, column].plot(
                    metric_df["epoch"],
                    metric_df["value"],
                    linewidth=1.5,
                    label=f"Fold {fold}",
                )
        axes[row, 0].set_title(f"{experiment} - Train loss")
        axes[row, 1].set_title(f"{experiment} - Validation loss")
        for column in range(2):
            axes[row, column].set_xlabel("Epoch")
            axes[row, column].set_ylabel("Cross-entropy loss")
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend(frameon=False, fontsize=7, ncol=5)

    plt.tight_layout()
    plt.savefig(output_folds_png, bbox_inches="tight")
    plt.close()

    colors = plt.get_cmap("tab10").colors
    _, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    for color, experiment in zip(colors, experiments):
        experiment_df = history_df[history_df["experiment"] == experiment]
        for axis, metric, title in zip(
            axes,
            ["train_loss", "eval_loss"],
            ["Mean Train Loss Across 5 Folds", "Mean Validation Loss Across 5 Folds"],
        ):
            metric_df = experiment_df[experiment_df["metric"] == metric]
            summary = metric_df.groupby("epoch")["value"].agg(["mean", "std", "count"])
            # Avoid a biased tail containing only folds that stopped later.
            summary = summary[summary["count"] == 5]
            epochs = summary.index.to_numpy(dtype=float)
            means = summary["mean"].to_numpy(dtype=float)
            stds = summary["std"].to_numpy(dtype=float)
            axis.plot(epochs, means, color=color, linewidth=2, label=experiment)
            axis.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.12)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Cross-entropy loss")
            axis.grid(alpha=0.2)

    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_mean_png, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    score_map = {}
    metric_rows = []

    for model_name, file_name in MODEL_FILES.items():
        file_path = SCORES_DIR / file_name
        if not file_path.exists():
            print(f"Skip {model_name}: {file_name} not found.")
            continue

        df = load_score_file(file_path)
        score_map[model_name] = df
        metrics = compute_metrics(df["label"].to_numpy(), df["score"].to_numpy(), THRESHOLD)
        metric_rows.append({"model": model_name, **metrics})

    if not metric_rows:
        raise FileNotFoundError(f"No score files found in {SCORES_DIR}")

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(FIGURES_DIR / "model_metrics_summary.csv", index=False)

    plot_roc_curves(score_map, metrics_df, FIGURES_DIR / "roc_curves.png")
    plot_pr_curves(score_map, metrics_df, FIGURES_DIR / "pr_curves.png")
    plot_metric_bars(metrics_df, FIGURES_DIR / "model_comparison_bar.png")
    plot_esm2_loss_curves(FIGURES_DIR / "esm2_loss_curves.png", FIGURES_DIR / "esm2_loss_history.csv")
    plot_esm2_ablation_losses(
        FIGURES_DIR / "esm2_ablation_loss_curves.png",
        FIGURES_DIR / "esm2_ablation_loss_mean_std.png",
        FIGURES_DIR / "esm2_ablation_loss_history.csv",
    )

    print("Saved figures to:", FIGURES_DIR)
    for file_name in [
        "roc_curves.png",
        "pr_curves.png",
        "model_comparison_bar.png",
        "model_metrics_summary.csv",
        "esm2_loss_curves.png",
        "esm2_loss_history.csv",
        "esm2_ablation_loss_curves.png",
        "esm2_ablation_loss_mean_std.png",
        "esm2_ablation_loss_history.csv",
    ]:
        print("-", FIGURES_DIR / file_name)


if __name__ == "__main__":
    main()
