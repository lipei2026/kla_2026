from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "predict" / "znf800_predictions.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "ZNF800_lollipop_metamodel.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ZNF800 Kla-site prediction scores.")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--score-column", default="MetaModel_score")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--title", default="ZNF800 Predicted Kla Sites by MetaModel")
    parser.add_argument("--x-label", default="ZNF800 Residue Position")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_excel(input_path) if input_path.suffix.lower() == ".xlsx" else pd.read_csv(input_path)
    if args.score_column not in df.columns:
        raise ValueError(f"Missing score column: {args.score_column}")
    df = df.sort_values("Site").reset_index(drop=True)

    protein_length = int(df["Protein_Length"].iloc[0]) if "Protein_Length" in df else int(df["Site"].max())
    top_df = df.nlargest(args.top_n, args.score_column).sort_values("Site")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.hlines(0, xmin=1, xmax=protein_length, color="#555555", linewidth=2.2, zorder=1)

    above_threshold = df[args.score_column] >= args.threshold
    colors = ["#D62728" if is_high else "#4C78A8" for is_high in above_threshold]

    ax.vlines(df["Site"], ymin=0, ymax=df[args.score_column], color=colors, linewidth=2.2, alpha=0.9, zorder=2)
    ax.scatter(df["Site"], df[args.score_column], s=70, c=colors, edgecolors="white", linewidths=0.8, zorder=3)

    ax.axhline(
        args.threshold,
        color="#F58518",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold = {args.threshold:.2f}",
    )

    label_offsets = {
        476: (-22, 12),
        482: (0, 22),
        489: (22, 12),
        616: (0, 12),
    }

    # Automatically spread labels for nearby sites. Explicit offsets above
    # preserve the original ZNF800 layout, while making this plotter reusable
    # for other proteins such as O15156/ZBTB7B.
    dynamic_offsets = {}
    clusters = []
    cluster = []
    for site in top_df["Site"].astype(int).tolist():
        if cluster and site - cluster[-1] > 15:
            clusters.append(cluster)
            cluster = []
        cluster.append(site)
    if cluster:
        clusters.append(cluster)
    for sites in clusters:
        for index, site in enumerate(sites):
            dx = (index - (len(sites) - 1) / 2) * 30
            dy = 12 + (index % 2) * 14
            dynamic_offsets[site] = (dx, dy)

    for _, row in top_df.iterrows():
        x = int(row["Site"])
        y = float(row[args.score_column])
        dx, dy = label_offsets.get(x, dynamic_offsets.get(x, (0, 12)))
        ax.annotate(
            f"K{x}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.8},
        )

    ax.set_xlim(0, protein_length + 10)
    ax.set_ylim(0, max(1.02, df[args.score_column].max() + 0.08))
    ax.set_xlabel(args.x_label, fontsize=12)
    ax.set_ylabel(args.score_column.replace("_", " "), fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.legend(frameon=False, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
