from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "predict" / "lab_external_test_predictions.xlsx"
DEFAULT_FIGURE = PROJECT_ROOT / "figures" / "lab_metamodel_score_distribution.png"
DEFAULT_LOW_SCORE_XLSX = PROJECT_ROOT / "predict" / "lab_low_score_samples.xlsx"
DEFAULT_LOW_SCORE_CSV = PROJECT_ROOT / "predict" / "lab_low_score_samples.csv"


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input file must be .xlsx, .xls or .csv")


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError("Output file must be .xlsx, .xls or .csv")


def plot_distribution(scores: pd.Series, output_path: Path, threshold: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [4, 1]})

    axes[0].hist(scores, bins=30, color="#4C78A8", edgecolor="white", alpha=0.9)
    axes[0].axvline(0.5, color="#F58518", linestyle="--", linewidth=2, label="Threshold = 0.5")
    axes[0].axvline(0.7, color="#54A24B", linestyle="--", linewidth=2, label="Threshold = 0.7")
    axes[0].axvline(0.9, color="#E45756", linestyle="--", linewidth=2, label="Threshold = 0.9")
    axes[0].axvline(scores.mean(), color="#B279A2", linestyle="-", linewidth=2, label=f"Mean = {scores.mean():.3f}")
    axes[0].axvline(scores.median(), color="#72B7B2", linestyle="-.", linewidth=2, label=f"Median = {scores.median():.3f}")
    axes[0].set_title("MetaModel Score Distribution on Lab Positive Samples")
    axes[0].set_xlabel("MetaModel Score")
    axes[0].set_ylabel("Sample Count")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].boxplot(scores, vert=True, patch_artist=True, boxprops={"facecolor": "#4C78A8", "alpha": 0.7})
    axes[1].axhline(threshold, color="#F58518", linestyle="--", linewidth=2)
    axes[1].set_title("Boxplot")
    axes[1].set_ylabel("MetaModel Score")
    axes[1].set_xticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MetaModel score distribution and export low-score lab samples.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Prediction table from run_lab_external_test.py")
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE), help="Output figure path (.png)")
    parser.add_argument("--low-score-xlsx", default=str(DEFAULT_LOW_SCORE_XLSX), help="Low-score sample output (.xlsx)")
    parser.add_argument("--low-score-csv", default=str(DEFAULT_LOW_SCORE_CSV), help="Low-score sample output (.csv)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used to define low-score positives")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = load_table(input_path)
    if "MetaModel_score" not in df.columns:
        raise ValueError("Input file must contain 'MetaModel_score'")

    df = df.copy()
    df["MetaModel_score"] = pd.to_numeric(df["MetaModel_score"], errors="coerce")
    df = df.dropna(subset=["MetaModel_score"]).reset_index(drop=True)

    low_score_df = df[df["MetaModel_score"] < args.threshold].copy()
    low_score_df = low_score_df.sort_values("MetaModel_score", ascending=True).reset_index(drop=True)

    save_table(low_score_df, Path(args.low_score_xlsx))
    save_table(low_score_df, Path(args.low_score_csv))
    plot_distribution(df["MetaModel_score"], Path(args.figure), args.threshold)

    print(f"Total samples: {len(df)}")
    print(f"Low-score samples (< {args.threshold}): {len(low_score_df)}")
    print(f"Saved low-score xlsx: {args.low_score_xlsx}")
    print(f"Saved low-score csv: {args.low_score_csv}")
    print(f"Saved figure: {args.figure}")


if __name__ == "__main__":
    main()
