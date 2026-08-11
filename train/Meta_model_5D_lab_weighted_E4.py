"""Train a separate 5D MetaModel with lab-adapted ESM2-E4 scores.

This script deliberately does not import or modify ``Meta_model_5D.py``.
Architecture, optimizer, fixed protein folds, seeds, and training procedure are
inherited from ``Meta_model_7D.py``; only the five selected input files and the
output experiment tag differ.

Required adapted E4 input:
    scores/ESM2_E4_lab_weighted_reference_scores.csv

Generate it first with:
    python generate_lab_weighted_e4_scores.py \
      --input train_data_with_ids.xlsx \
      --output scores/ESM2_E4_lab_weighted_reference_scores.csv \
      --keep-duplicate-sites \
      --resolved-folds-only

Important: the E4 heads were adapted with laboratory labels. Consequently,
metrics from this MetaModel are not a clean replacement for the original
protein-grouped CV baseline if laboratory proteins overlap the reference data.
Use a separately frozen external set for the final domain-adaptation claim.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

import Meta_model_7D as meta


SCRIPT_DIR = Path(__file__).resolve().parent
SCORES_DIR = SCRIPT_DIR / "scores"
ADAPTED_E4_FILE = "ESM2_E4_lab_weighted_reference_scores.csv"
EXPERIMENT_TAG = "E4folds_grouped_5D_lab_weighted_E4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a new 5D MetaModel using lab-weighted E4 scores."
    )
    parser.add_argument(
        "--adapted-e4-file",
        default=ADAPTED_E4_FILE,
        help="Filename inside train/scores containing the adapted E4 scores.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all five score files and their alignment without training.",
    )
    return parser.parse_args()


def selected_score_files(adapted_e4_file: str) -> dict[str, str]:
    if Path(adapted_e4_file).name != adapted_e4_file:
        raise ValueError("--adapted-e4-file must be a filename inside train/scores")
    return {
        "E4_lab_weighted": adapted_e4_file,
        "LSTM": f"LSTM_E4folds_grouped_y_label&score{meta.NUM_FOLDS}.csv",
        "AAINDEX": f"AAINDEX_E4folds_grouped_y_label&score{meta.NUM_FOLDS}.csv",
        "CKSAAP": f"CKSAAP_E4folds_grouped_y_label&score{meta.NUM_FOLDS}.csv",
        "OBC": f"OBC_E4folds_grouped_y_label&score{meta.NUM_FOLDS}.csv",
    }


def load_and_validate_inputs(files: dict[str, str]) -> None:
    keys = ["Sample_ID", "Group_ID", "Fold"]
    reference_rows: pd.DataFrame | None = None

    for feature, filename in files.items():
        path = SCORES_DIR / filename
        if not path.is_file():
            if feature == "E4_lab_weighted":
                raise FileNotFoundError(
                    f"Missing adapted E4 score file: {path}\n"
                    "Generate it with:\n"
                    "  .venv/bin/python train/generate_lab_weighted_e4_scores.py "
                    "--input train/train_data_with_ids.xlsx "
                    "--output train/scores/ESM2_E4_lab_weighted_reference_scores.csv "
                    "--keep-duplicate-sites --resolved-folds-only"
                )
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        required = {*keys, "label", "score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        if len(df) != 46403:
            raise ValueError(f"{filename} has {len(df)} rows; expected 46,403")
        if df.duplicated(keys).any():
            raise ValueError(f"{filename} contains duplicate Sample_ID/Group_ID/Fold keys")
        if not df["Fold"].isin(range(1, meta.NUM_FOLDS + 1)).all():
            raise ValueError(f"{filename} contains folds outside 1-5")
        if not df["score"].between(0, 1).all():
            raise ValueError(f"{filename} contains scores outside [0, 1]")

        current_rows = df[keys + ["label"]].copy()
        current_rows["label"] = current_rows["label"].astype(int)
        current_rows = current_rows.sort_values(keys).reset_index(drop=True)
        if reference_rows is None:
            reference_rows = current_rows
        else:
            if not current_rows.equals(reference_rows):
                raise ValueError(
                    f"{filename} sample keys or labels do not match adapted E4"
                )

        if feature == "E4_lab_weighted":
            if "score_strategy" not in df.columns:
                raise ValueError(
                    f"{filename} lacks score_strategy; regenerate it with the new E4 scorer"
                )
            if not df["score_strategy"].eq("fixed_fold_aligned").all():
                counts = df["score_strategy"].value_counts().to_dict()
                raise ValueError(
                    "Adapted E4 reference scores must all be fixed_fold_aligned; "
                    f"found {counts}"
                )
        print(f"Validated {feature}: {len(df)} aligned rows from {filename}")


def main() -> None:
    args = parse_args()
    files = selected_score_files(args.adapted_e4_file)
    load_and_validate_inputs(files)
    if args.dry_run:
        print("Dry run complete; no MetaModel was trained or written.")
        return

    # The inherited implementation uses paths relative to train/.
    os.chdir(SCRIPT_DIR)
    meta.BASE_SCORE_FILES = files
    meta.SPLIT_TAG = EXPERIMENT_TAG
    meta.FOLD_REFERENCE_PATH = "./train_data_with_ids.xlsx"
    meta.set_seed(meta.SEED)

    fold = meta.NUM_FOLDS
    (
        dataset,
        feature_scores,
        feature_labels,
        labels,
        sample_ids,
        groups,
        folds,
    ) = meta.auc_11("./scores", fold)
    if dataset.shape != (46403, 5):
        raise RuntimeError(f"Expected a (46403, 5) MetaModel matrix, found {dataset.shape}")
    print(f"Training independent adapted-E4 5D MetaModel with inputs: {list(files)}")

    _, feature_scores, feature_labels = meta.training_DNN(
        sample_ids,
        groups,
        folds,
        dataset,
        labels,
        feature_scores,
        feature_labels,
        fold,
    )
    meta.roc(feature_labels, feature_scores)
    print(
        "Saved predictions to "
        f"scores/meta_model_{EXPERIMENT_TAG}_y_label&score{fold}.csv"
    )


if __name__ == "__main__":
    main()
