"""Generate E4 scores with the lab-weighted classification heads.

This is inference only: it loads the five checkpoints produced by
``finetune_lab_e4_weighted.py`` and does not update any parameter.

For input without a ``Fold`` column, the canonical ``score`` is the mean of
the five adapted models.  For input carrying fixed folds 1-5, the canonical
``score`` is selected from the matching fold model so downstream files remain
fold-aligned.  Per-fold and ensemble scores are always retained.

The adapted E4 score distribution differs from the original E4 distribution.
Do not silently overwrite the original E4 OOF file or feed these scores into
the existing 5D MetaModel without retraining/recalibrating that MetaModel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICT_DIR = PROJECT_ROOT / "predict"
sys.path.insert(0, str(PREDICT_DIR))

from predict_znf800_e4 import (  # noqa: E402
    CENTER_RESIDUE_INDEX,
    ESM2CenterKMeanClassifier,
    WINDOW_LENGTH,
    fold_number,
    tokenize_windows,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "train_data_lab.xlsx"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "train" / "trained_models" / "ESM2_lab_weighted"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "train" / "scores" / "E4_lab_weighted_lab_scores.xlsx"
)
MODEL_PATTERN = "E4_lab_weighted_fold_*_5"
NUM_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate inference scores with the five lab-weighted E4 heads."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--keep-duplicate-sites",
        action="store_true",
        help=(
            "Keep repeated UniProt_ID/Site rows. By default, site-key duplicates "
            "are removed to match finetune_lab_e4_weighted.py."
        ),
    )
    parser.add_argument(
        "--resolved-folds-only",
        action="store_true",
        help=(
            "Keep only rows whose Fold is 1-5. Use this with "
            "--keep-duplicate-sites when regenerating the 46,403-row reference "
            "score file for a fold-aligned MetaModel."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. 'auto' selects CUDA when available.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be an .xlsx, .xls, or .csv file")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output must be an .xlsx or .csv file")


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    if "Sequence" not in df.columns:
        raise ValueError("Input must contain a Sequence column")
    result = df.copy()
    result["Sequence"] = result["Sequence"].astype(str).str.upper()
    valid_length = result["Sequence"].str.len().eq(WINDOW_LENGTH)
    if not valid_length.all():
        bad_rows = result.index[~valid_length].tolist()[:10]
        raise ValueError(
            f"All windows must contain {WINDOW_LENGTH} residues; bad rows: {bad_rows}"
        )
    valid_center = result["Sequence"].str[CENTER_RESIDUE_INDEX].eq("K")
    if not valid_center.all():
        bad_rows = result.index[~valid_center].tolist()[:10]
        raise ValueError(f"Every window must have center K; bad rows: {bad_rows}")
    return result


def deduplicate_site_keys(df: pd.DataFrame) -> pd.DataFrame:
    keys = {"UniProt_ID", "Site"}
    if not keys.issubset(df.columns):
        print("No UniProt_ID/Site columns; preserving all input rows")
        return df
    result = df.copy()
    result["Site"] = pd.to_numeric(result["Site"], errors="raise").astype(int)
    if "Label" in result.columns:
        conflicts = result.groupby(["UniProt_ID", "Site"], dropna=False)[
            "Label"
        ].nunique()
        if (conflicts > 1).any():
            raise ValueError("Contradictory labels found for duplicate site keys")
    before = len(result)
    result = result.drop_duplicates(["UniProt_ID", "Site"], keep="first").reset_index(
        drop=True
    )
    print(f"Deduplicated input site keys: {before} -> {len(result)} rows")
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_models(model_dir: Path) -> dict[int, Path]:
    models = {
        fold_number(path): path
        for path in model_dir.glob(MODEL_PATTERN)
        if path.is_dir()
    }
    expected = set(range(1, NUM_FOLDS + 1))
    if set(models) != expected:
        raise FileNotFoundError(
            f"Expected adapted E4 folds 1-5 in {model_dir}; found {sorted(models)}"
        )
    return models


def predict_one_fold(
    model_path: Path,
    tokenized: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = ESM2CenterKMeanClassifier.from_pretrained(model_path).to(device)
    model.eval()
    scores: list[np.ndarray] = []
    total = tokenized["input_ids"].shape[0]
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            logits = model(
                input_ids=tokenized["input_ids"][start:stop].to(device),
                attention_mask=tokenized["attention_mask"][start:stop].to(device),
            ).logits
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(scores).astype(float)


def add_canonical_score(
    df: pd.DataFrame,
    fold_scores: dict[int, np.ndarray],
    ensemble_score: np.ndarray,
) -> tuple[pd.DataFrame, str]:
    result = df.copy()
    if "Fold" not in result.columns:
        result["score"] = ensemble_score
        result["score_strategy"] = "five_fold_ensemble_mean"
        return result, "five_fold_ensemble_mean"

    numeric_folds = pd.to_numeric(result["Fold"], errors="coerce")
    valid_fold = numeric_folds.isin(range(1, NUM_FOLDS + 1))
    fold_aligned = np.full(len(result), np.nan, dtype=float)
    for fold, scores in fold_scores.items():
        mask = numeric_folds.eq(fold).to_numpy()
        fold_aligned[mask] = scores[mask]

    result["E4_lab_weighted_fold_aligned_score"] = fold_aligned
    # Fold 0/unassigned rows cannot have an OOF-aligned model, so retain an
    # ensemble score for inference while marking the strategy explicitly.
    result["score"] = np.where(valid_fold, fold_aligned, ensemble_score)
    result["score_strategy"] = np.where(
        valid_fold,
        "fixed_fold_aligned",
        "five_fold_ensemble_mean_unassigned",
    )
    return result, "fixed_fold_aligned_with_ensemble_for_unassigned"


def write_summary(
    df: pd.DataFrame,
    output_path: Path,
    strategy: str,
    threshold: float,
) -> Path:
    summary_path = output_path.with_suffix(".summary.txt")
    label_column = "Label" if "Label" in df.columns else (
        "label" if "label" in df.columns else None
    )
    lines = [
        "Lab-weighted E4 inference summary",
        f"rows: {len(df)}",
        f"score_strategy: {strategy}",
        f"threshold: {threshold:.2f}",
        f"mean_score: {df['score'].mean():.6f}",
        f"median_score: {df['score'].median():.6f}",
        f"predicted_positive: {int(df['prediction'].sum())}",
        f"predicted_positive_rate: {df['prediction'].mean():.6f}",
    ]
    if label_column is not None:
        labels = pd.to_numeric(df[label_column], errors="coerce")
        positive = labels.eq(1)
        unlabeled = labels.eq(0)
        if positive.any():
            lines.extend(
                [
                    f"confirmed_positive_count: {int(positive.sum())}",
                    "confirmed_positive_recall: "
                    f"{df.loc[positive, 'prediction'].mean():.6f}",
                    "confirmed_positive_mean_score: "
                    f"{df.loc[positive, 'score'].mean():.6f}",
                ]
            )
        if unlabeled.any():
            lines.extend(
                [
                    f"unlabeled_count: {int(unlabeled.sum())}",
                    "unlabeled_positive_rate: "
                    f"{df.loc[unlabeled, 'prediction'].mean():.6f}",
                    "unlabeled_mean_score: "
                    f"{df.loc[unlabeled, 'score'].mean():.6f}",
                    "note: Label=0 rows are unlabeled K sites, not verified negatives",
                ]
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    df = read_table(args.input)
    if args.resolved_folds_only:
        if "Fold" not in df.columns:
            raise ValueError("--resolved-folds-only requires a Fold column")
        folds = pd.to_numeric(df["Fold"], errors="coerce")
        before = len(df)
        df = df.loc[folds.isin(range(1, NUM_FOLDS + 1))].reset_index(drop=True)
        print(f"Filtered to resolved folds 1-5: {before} -> {len(df)} rows")
    if not args.keep_duplicate_sites:
        df = deduplicate_site_keys(df)
    df = validate_input(df)
    models = discover_models(args.model_dir)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(models[1])
    tokenized = tokenize_windows(tokenizer, df["Sequence"].tolist())
    print(f"Scoring {len(df)} windows on {device} with {NUM_FOLDS} adapted E4 heads")

    fold_scores: dict[int, np.ndarray] = {}
    for fold in range(1, NUM_FOLDS + 1):
        print(f"Fold {fold}/{NUM_FOLDS}: {models[fold].name}")
        fold_scores[fold] = predict_one_fold(
            models[fold],
            tokenized,
            device,
            args.batch_size,
        )
        df[f"E4_lab_weighted_fold{fold}_score"] = fold_scores[fold]

    score_matrix = np.vstack([fold_scores[fold] for fold in range(1, NUM_FOLDS + 1)])
    ensemble_score = score_matrix.mean(axis=0)
    df["E4_lab_weighted_ensemble_score"] = ensemble_score
    df["E4_lab_weighted_ensemble_std"] = score_matrix.std(axis=0, ddof=1)
    df, strategy = add_canonical_score(df, fold_scores, ensemble_score)
    df["prediction"] = (df["score"] >= args.threshold).astype(int)
    if "Label" in df.columns and "label" not in df.columns:
        df["label"] = pd.to_numeric(df["Label"], errors="raise").astype(int)

    write_table(df, args.output)
    summary_path = write_summary(df, args.output, strategy, args.threshold)
    print(f"Saved scores to {args.output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
