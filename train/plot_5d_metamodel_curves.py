"""Plot ROC and precision-recall curves for the ESM2-E4 5D MetaModel.

The comparison contains the five out-of-fold inputs used by the MetaModel and
the final out-of-fold MetaModel predictions. All files use the shared E4
protein-level folds.
"""

from pathlib import Path

import pandas as pd

from plot_model_figures import (
    THRESHOLD,
    compute_metrics,
    load_score_file,
    plot_pr_curves,
    plot_roc_curves,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "train" / "scores"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODEL_FILES = {
    "ESM2-E4": "ESM2_E4_center_k_mean_grouped_y_label&score5.csv",
    "LSTM": "LSTM_E4folds_grouped_y_label&score5.csv",
    "AAINDEX": "AAINDEX_E4folds_grouped_y_label&score5.csv",
    "CKSAAP": "CKSAAP_E4folds_grouped_y_label&score5.csv",
    "OBC": "OBC_E4folds_grouped_y_label&score5.csv",
    "5D MetaModel": "meta_model_E4folds_grouped_5D_y_label&score5.csv",
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    score_map = {}
    metric_rows = []
    reference_labels = None

    for model_name, file_name in MODEL_FILES.items():
        file_path = SCORES_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required score file not found: {file_path}")

        df = load_score_file(file_path)
        labels = df["label"].to_numpy()
        if reference_labels is None:
            reference_labels = labels
        elif len(labels) != len(reference_labels) or (labels != reference_labels).any():
            raise ValueError(f"Labels are not aligned in {file_name}")

        score_map[model_name] = df
        metrics = compute_metrics(labels, df["score"].to_numpy(), THRESHOLD)
        metric_rows.append({"model": model_name, **metrics})

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    metrics_path = FIGURES_DIR / "esm2_e4_5d_metamodel_metrics.csv"
    roc_path = FIGURES_DIR / "esm2_e4_5d_metamodel_roc_curves.png"
    pr_path = FIGURES_DIR / "esm2_e4_5d_metamodel_pr_curves.png"

    metrics_df.to_csv(metrics_path, index=False)
    plot_roc_curves(score_map, metrics_df, roc_path)
    plot_pr_curves(score_map, metrics_df, pr_path)

    print(metrics_df.to_string(index=False))
    print(f"Saved ROC curve: {roc_path}")
    print(f"Saved PR curve:  {pr_path}")
    print(f"Saved metrics:   {metrics_path}")


if __name__ == "__main__":
    main()
