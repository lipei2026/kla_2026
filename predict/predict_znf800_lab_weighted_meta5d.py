"""Predict ZNF800 with lab-adapted E4 and its independently retrained 5D MetaModel.

The original ZNF800 E4/5D prediction scripts and outputs are not modified.
For each fold, this script combines:
    lab-weighted E4, LSTM, AAINDEX, CKSAAP, OBC
and applies the matching lab-weighted-E4 5D MetaModel before averaging the
five fold-level MetaModel predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import predict as predict_module
from predict_znf800_e4 import tokenize_windows
from predict_znf800_meta5d import (
    NUM_FOLDS,
    aaindex_encode,
    predict_e4_fold,
    predict_feature_fold,
    predict_lstm_fold,
    predict_meta_fold,
    require_file,
    validate_candidates,
)
from run_lab_external_test import kmors_encode, one_hot_binary_encode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "train" / "trained_models"
ADAPTED_E4_DIR = TRAINED_MODELS_DIR / "ESM2_lab_weighted"
DEFAULT_INPUT = PROJECT_ROOT / "predict" / "znf800_candidates.xlsx"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "predict" / "znf800_LabWeightedE4_Meta5D_predictions.xlsx"
)
THRESHOLD = 0.5
META_INPUT_ORDER = ["E4_lab_weighted", "LSTM", "AAINDEX", "CKSAAP", "OBC"]
META_EXPERIMENT_TAG = "E4folds_grouped_5D_lab_weighted_E4"


def adapted_e4_models() -> dict[int, Path]:
    models = {
        fold: ADAPTED_E4_DIR / f"E4_lab_weighted_fold_{fold}_5"
        for fold in range(1, NUM_FOLDS + 1)
    }
    missing = [path for path in models.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing adapted E4 model directories: {missing}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one protein using lab-weighted E4 and the corresponding new 5D MetaModel."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protein-name", default="ZNF800")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    raw = (
        pd.read_excel(args.input)
        if args.input.suffix.lower() == ".xlsx"
        else pd.read_csv(args.input)
    )
    df = validate_candidates(raw)
    sequences = df["Sequence"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e4_models = adapted_e4_models()
    tokenizer = predict_module.AutoTokenizer.from_pretrained(e4_models[1])
    tokenized = tokenize_windows(tokenizer, sequences)
    lstm_tensor = torch.tensor(
        predict_module.seq2num(sequences),
        dtype=torch.long,
    )
    feature_tensors = {
        "AAINDEX": torch.tensor(
            np.asarray(aaindex_encode(sequences)),
            dtype=torch.float32,
        ),
        "CKSAAP": torch.tensor(
            np.asarray(kmors_encode(sequences, 1, mode="l")),
            dtype=torch.float32,
        ),
        "OBC": torch.tensor(
            np.asarray(one_hot_binary_encode(sequences, include_u=0)),
            dtype=torch.float32,
        ),
    }
    for name, tensor in feature_tensors.items():
        expected = predict_module.size_dist[name]
        if tensor.shape != (len(df), expected):
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match {(len(df), expected)}"
            )

    print(
        f"Predicting {len(df)} {args.protein_name} candidate K sites on {device}; "
        f"input order: {META_INPUT_ORDER}"
    )
    meta_fold_scores: list[np.ndarray] = []
    base_scores_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in META_INPUT_ORDER
    }

    for fold in range(1, NUM_FOLDS + 1):
        print(f"Fold {fold}/{NUM_FOLDS}")
        fold_scores: dict[str, np.ndarray] = {}
        fold_scores["E4_lab_weighted"] = predict_e4_fold(
            e4_models[fold],
            tokenized,
            device,
            args.batch_size,
        )
        lstm_path = require_file(
            TRAINED_MODELS_DIR
            / "LSTM"
            / f"fold_{fold}_5_E4folds_grouped_lstm.pth"
        )
        fold_scores["LSTM"] = predict_lstm_fold(
            lstm_path,
            lstm_tensor,
            device,
            args.batch_size,
        )
        for feature_name in ["AAINDEX", "CKSAAP", "OBC"]:
            feature_path = require_file(
                TRAINED_MODELS_DIR
                / feature_name
                / f"{feature_name}{fold}_5_E4folds_grouped_DNN_.pth"
            )
            fold_scores[feature_name] = predict_feature_fold(
                feature_name,
                feature_path,
                feature_tensors[feature_name],
                device,
            )

        stacked = np.column_stack(
            [fold_scores[name] for name in META_INPUT_ORDER]
        )
        meta_path = require_file(
            TRAINED_MODELS_DIR
            / f"meta_model_{META_EXPERIMENT_TAG}_fold_{fold}_{NUM_FOLDS}.pth"
        )
        meta_scores = predict_meta_fold(meta_path, stacked, device)
        meta_fold_scores.append(meta_scores)
        df[f"LabWeighted_Meta5D_fold{fold}_score"] = meta_scores
        for name in META_INPUT_ORDER:
            base_scores_by_name[name].append(fold_scores[name])
            df[f"{name}_fold{fold}_score"] = fold_scores[name]

    meta_matrix = np.vstack(meta_fold_scores)
    df["LabWeighted_Meta5D_score"] = meta_matrix.mean(axis=0)
    df["LabWeighted_Meta5D_score_std"] = meta_matrix.std(axis=0, ddof=1)
    df["LabWeighted_Meta5D_prediction"] = (
        df["LabWeighted_Meta5D_score"] >= args.threshold
    ).astype(int)
    for name, scores in base_scores_by_name.items():
        df[f"{name}_score"] = np.vstack(scores).mean(axis=0)

    df = df.sort_values(
        "LabWeighted_Meta5D_score",
        ascending=False,
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".xlsx":
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    positive = df[df["LabWeighted_Meta5D_prediction"].eq(1)]
    summary_path = args.output.with_suffix(".summary.txt")
    lines = [
        f"{args.protein_name} prediction with lab-weighted E4 and retrained 5D MetaModel",
        f"input_order: {', '.join(META_INPUT_ORDER)}",
        f"threshold: {args.threshold:.2f}",
        f"candidate_K_sites: {len(df)}",
        f"predicted_positive_sites: {len(positive)}",
        f"mean_score: {df['LabWeighted_Meta5D_score'].mean():.6f}",
        f"max_score: {df['LabWeighted_Meta5D_score'].max():.6f}",
    ]
    if len(positive):
        ranked = ", ".join(f"K{int(site)}" for site in positive["Site"])
        lines.append(f"positive_sites_ranked: {ranked}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved predictions to {args.output}")
    print(f"Saved summary to {summary_path}")
    print(
        df[
            [
                "Site",
                "LabWeighted_Meta5D_score",
                "LabWeighted_Meta5D_score_std",
                "LabWeighted_Meta5D_prediction",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
