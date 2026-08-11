"""Weighted lab-domain adaptation for the E4 ESM2 classifier head.

The laboratory table contains confirmed positive Kla sites and candidate lysines
that were not observed as positive.  The latter are treated as *unlabeled*, not
as equally reliable negatives.  The default loss weights are therefore:

    confirmed lab positive: 2.0
    lab unlabeled K site:    0.2

Only the E4 classification head is optimized.  The ESM2 backbone is frozen and
kept in evaluation mode.  By default, laboratory proteins absent from the
original training reference are excluded from adaptation and used as a frozen
positive-only benchmark.

This script creates adapted E4 checkpoints.  Existing 5D MetaModel weights were
trained on the original E4 score distribution and must not be combined with the
adapted E4 scores without retraining/recalibrating the MetaModel.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICT_DIR = PROJECT_ROOT / "predict"
sys.path.insert(0, str(PREDICT_DIR))

from predict_znf800_e4 import (  # noqa: E402
    CENTER_RESIDUE_INDEX,
    E4_MODEL_DIR,
    E4_MODEL_PATTERN,
    ESM2CenterKMeanClassifier,
    WINDOW_LENGTH,
    fold_number,
    tokenize_windows,
)


DEFAULT_LAB_DATA = PROJECT_ROOT / "data" / "train_data_lab.xlsx"
DEFAULT_REFERENCE_DATA = PROJECT_ROOT / "train" / "train_data_with_ids.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "train" / "trained_models" / "ESM2_lab_weighted"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "train" / "scores" / "E4_lab_weighted_finetune_metrics.csv"


@dataclass
class Evaluation:
    loss: float
    positive_count: int
    unlabeled_count: int
    positive_recall_at_threshold: float | None
    unlabeled_positive_rate_at_threshold: float | None
    mean_positive_score: float | None
    mean_unlabeled_score: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze ESM2 and adapt the E4 classification head to weighted lab data."
    )
    parser.add_argument("--lab-data", type=Path, default=DEFAULT_LAB_DATA)
    parser.add_argument("--reference-data", type=Path, default=DEFAULT_REFERENCE_DATA)
    parser.add_argument("--model-dir", type=Path, default=E4_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--unlabeled-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Original E4 folds to adapt.",
    )
    parser.add_argument(
        "--use-all-lab-proteins",
        action="store_true",
        help=(
            "Also adapt on proteins absent from the original training reference. "
            "This consumes the default frozen new-protein benchmark."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print the protein-level split without loading a model.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def validate_windows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    result = df.copy()
    result["Sequence"] = result["Sequence"].astype(str).str.upper()
    if not result["Sequence"].str.len().eq(WINDOW_LENGTH).all():
        bad = result.loc[~result["Sequence"].str.len().eq(WINDOW_LENGTH)].head()
        raise ValueError(f"{name} contains non-{WINDOW_LENGTH}-residue windows:\n{bad}")
    if not result["Sequence"].str[CENTER_RESIDUE_INDEX].eq("K").all():
        bad = result.loc[
            ~result["Sequence"].str[CENTER_RESIDUE_INDEX].eq("K")
        ].head()
        raise ValueError(f"{name} contains windows without center K:\n{bad}")
    return result


def deduplicate_lab_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate site keys and reject contradictory labels."""
    result = df.copy()
    result["Site"] = pd.to_numeric(result["Site"], errors="raise").astype(int)
    result["Label"] = pd.to_numeric(result["Label"], errors="raise").astype(int)
    if not set(result["Label"].unique()).issubset({0, 1}):
        raise ValueError("Lab labels must be 0 (unlabeled) or 1 (confirmed positive)")

    key = ["UniProt_ID", "Site"]
    conflicts = result.groupby(key, dropna=False)["Label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(
            "Contradictory lab labels for site keys: "
            f"{list(conflicts.index[:10])}"
        )

    before = len(result)
    result = result.drop_duplicates(key, keep="first").reset_index(drop=True)
    print(f"Deduplicated lab site keys: {before} -> {len(result)} rows")
    return result


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, set[str]]:
    lab = pd.read_excel(args.lab_data)
    reference = pd.read_excel(args.reference_data)
    require_columns(lab, {"UniProt_ID", "Site", "Sequence", "Label"}, "lab data")
    require_columns(reference, {"UniProt_ID", "Group_ID"}, "training reference")
    lab = validate_windows(deduplicate_lab_rows(lab), "lab data")
    if lab["UniProt_ID"].isna().any():
        raise ValueError("Every lab row must have a resolved UniProt_ID")

    seen_proteins = set(reference["UniProt_ID"].dropna().astype(str))
    seen_proteins.update(reference["Group_ID"].dropna().astype(str))
    return lab, seen_proteins


def split_by_protein(
    lab: pd.DataFrame,
    seen_proteins: set[str],
    validation_fraction: float,
    seed: int,
    use_all_lab_proteins: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")

    positive_proteins = set(
        lab.loc[lab["Label"].eq(1), "UniProt_ID"].astype(str)
    )
    novel_proteins = positive_proteins - seen_proteins
    frozen = lab.loc[
        lab["Label"].eq(1) & lab["UniProt_ID"].astype(str).isin(novel_proteins)
    ].copy()

    if use_all_lab_proteins:
        adaptation = lab.copy()
        frozen = lab.iloc[0:0].copy()
    else:
        adaptation = lab.loc[
            ~lab["UniProt_ID"].astype(str).isin(novel_proteins)
        ].copy()

    proteins = sorted(adaptation["UniProt_ID"].astype(str).unique())
    if len(proteins) < 2:
        raise ValueError("Need at least two adaptation proteins for a grouped split")
    validation_count = min(
        len(proteins) - 1,
        max(1, round(len(proteins) * validation_fraction)),
    )
    # Protein sizes are highly uneven in this dataset.  A single random group
    # split can put far more than the requested fraction of sites into
    # validation.  Search deterministic random candidates and retain the split
    # closest to the requested row, positive, and unlabeled fractions.
    group_counts = adaptation.assign(
        _protein=adaptation["UniProt_ID"].astype(str)
    ).groupby("_protein")["Label"].agg(rows="size", positives="sum")
    total_rows = float(len(adaptation))
    total_positives = float(adaptation["Label"].sum())
    total_unlabeled = total_rows - total_positives
    rng = random.Random(seed)
    best_score = float("inf")
    validation_proteins: set[str] | None = None
    for _ in range(2000):
        candidate = set(rng.sample(proteins, validation_count))
        selected = group_counts.loc[list(candidate)]
        rows = float(selected["rows"].sum())
        positives = float(selected["positives"].sum())
        unlabeled = rows - positives
        score = (
            abs(rows / total_rows - validation_fraction)
            + abs(positives / max(total_positives, 1.0) - validation_fraction)
            + abs(unlabeled / max(total_unlabeled, 1.0) - validation_fraction)
        )
        if score < best_score:
            best_score = score
            validation_proteins = candidate
    if validation_proteins is None:
        raise RuntimeError("Unable to create a protein-group validation split")
    validation = adaptation.loc[
        adaptation["UniProt_ID"].astype(str).isin(validation_proteins)
    ].copy()
    train = adaptation.loc[
        ~adaptation["UniProt_ID"].astype(str).isin(validation_proteins)
    ].copy()

    train_proteins = set(train["UniProt_ID"].astype(str))
    validation_ids = set(validation["UniProt_ID"].astype(str))
    frozen_ids = set(frozen["UniProt_ID"].astype(str))
    if train_proteins & validation_ids:
        raise RuntimeError("Protein leakage between adaptation train and validation")
    if not use_all_lab_proteins and (train_proteins | validation_ids) & frozen_ids:
        raise RuntimeError("Frozen new-protein benchmark leaked into adaptation data")
    if not train["Label"].eq(1).any() or not validation["Label"].eq(1).any():
        raise ValueError("Both adaptation train and validation require positive samples")
    return train.reset_index(drop=True), validation.reset_index(drop=True), frozen.reset_index(drop=True)


def describe_split(name: str, df: pd.DataFrame) -> dict[str, int]:
    summary = {
        "rows": len(df),
        "proteins": int(df["UniProt_ID"].nunique()) if len(df) else 0,
        "positives": int(df["Label"].eq(1).sum()) if len(df) else 0,
        "unlabeled": int(df["Label"].eq(0).sum()) if len(df) else 0,
    }
    print(f"{name}: {summary}")
    return summary


def validate_split_accounting(
    lab: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    frozen: pd.DataFrame,
    use_all_lab_proteins: bool,
) -> dict[str, int]:
    """Verify positive conservation and account for intentionally omitted rows."""
    included = pd.concat([train, validation, frozen], ignore_index=True)
    total_positives = int(lab["Label"].eq(1).sum())
    included_positives = int(included["Label"].eq(1).sum())
    if included_positives != total_positives:
        raise RuntimeError(
            "Split accounting lost or duplicated confirmed positives: "
            f"{included_positives} included versus {total_positives} total"
        )

    omitted_rows = len(lab) - len(included)
    omitted_unlabeled = int(lab["Label"].eq(0).sum()) - int(
        included["Label"].eq(0).sum()
    )
    if omitted_rows < 0 or omitted_rows != omitted_unlabeled:
        raise RuntimeError(
            "Rows omitted by the split must consist only of unlabeled K sites"
        )
    if use_all_lab_proteins and omitted_rows:
        raise RuntimeError("--use-all-lab-proteins should not omit any lab rows")

    accounting = {
        "deduplicated_lab_rows": len(lab),
        "confirmed_positives_total": total_positives,
        "included_rows": len(included),
        "included_unlabeled": int(included["Label"].eq(0).sum()),
        "excluded_novel_protein_unlabeled": omitted_unlabeled,
    }
    print(f"split_accounting: {accounting}")
    return accounting


def build_loader(
    tokenizer,
    df: pd.DataFrame,
    positive_weight: float,
    unlabeled_weight: float,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tokenized = tokenize_windows(tokenizer, df["Sequence"].tolist())
    labels = torch.tensor(df["Label"].to_numpy(), dtype=torch.long)
    weights = torch.where(
        labels.eq(1),
        torch.full_like(labels, positive_weight, dtype=torch.float32),
        torch.full_like(labels, unlabeled_weight, dtype=torch.float32),
    )
    dataset = TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        labels,
        weights,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def evaluate(model, loader: DataLoader, device: torch.device, threshold: float) -> Evaluation:
    model.eval()
    total_weighted_loss = 0.0
    total_weight = 0.0
    labels_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []
    with torch.inference_mode():
        for input_ids, attention_mask, labels, weights in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            losses = F.cross_entropy(logits, labels, reduction="none")
            total_weighted_loss += float((losses * weights).sum().cpu())
            total_weight += float(weights.sum().cpu())
            labels_all.append(labels.cpu())
            scores_all.append(torch.softmax(logits, dim=-1)[:, 1].cpu())

    labels = torch.cat(labels_all)
    scores = torch.cat(scores_all)
    positive = labels.eq(1)
    unlabeled = labels.eq(0)

    def optional_mean(values: torch.Tensor) -> float | None:
        return float(values.mean()) if values.numel() else None

    positive_scores = scores[positive]
    unlabeled_scores = scores[unlabeled]
    return Evaluation(
        loss=total_weighted_loss / max(total_weight, 1e-12),
        positive_count=int(positive.sum()),
        unlabeled_count=int(unlabeled.sum()),
        positive_recall_at_threshold=optional_mean(
            (positive_scores >= threshold).float()
        ),
        unlabeled_positive_rate_at_threshold=optional_mean(
            (unlabeled_scores >= threshold).float()
        ),
        mean_positive_score=optional_mean(positive_scores),
        mean_unlabeled_score=optional_mean(unlabeled_scores),
    )


def train_one_fold(
    fold: int,
    source_dir: Path,
    tokenizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    frozen_loader: DataLoader | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    print(f"\nFold {fold}: loading {source_dir.name}")
    model = ESM2CenterKMeanClassifier.from_pretrained(source_dir).to(device)
    for parameter in model.esm.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Fold {fold}: trainable parameters={trainable:,}; frozen parameters={frozen:,}")

    before_validation = evaluate(model, validation_loader, device, args.threshold)
    before_frozen = (
        evaluate(model, frozen_loader, device, args.threshold)
        if frozen_loader is not None
        else None
    )

    optimizer = AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_classifier_state = copy.deepcopy(model.classifier.state_dict())
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.classifier.train()
        model.esm.eval()
        running_loss = 0.0
        batches = 0
        for input_ids, attention_mask, labels, weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.classifier.parameters(), args.max_grad_norm
            )
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

        validation = evaluate(model, validation_loader, device, args.threshold)
        print(
            f"Fold {fold} epoch {epoch}: train_loss={running_loss / max(batches, 1):.6f}; "
            f"val_loss={validation.loss:.6f}; "
            f"val_positive_recall={validation.positive_recall_at_threshold:.4f}; "
            "val_unlabeled_positive_rate="
            f"{validation.unlabeled_positive_rate_at_threshold:.4f}"
        )
        if validation.loss < best_loss - args.min_delta:
            best_loss = validation.loss
            best_epoch = epoch
            stale_epochs = 0
            best_classifier_state = copy.deepcopy(model.classifier.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Fold {fold}: early stopping at epoch {epoch}")
                break

    model.classifier.load_state_dict(best_classifier_state)
    after_validation = evaluate(model, validation_loader, device, args.threshold)
    after_frozen = (
        evaluate(model, frozen_loader, device, args.threshold)
        if frozen_loader is not None
        else None
    )

    output_dir = args.output_dir / f"E4_lab_weighted_fold_{fold}_5"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fold {fold}: saved best epoch {best_epoch} to {output_dir}")

    result: dict[str, object] = {
        "fold": fold,
        "source_model": str(source_dir),
        "output_model": str(output_dir),
        "best_epoch": best_epoch,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }
    for prefix, evaluation in [
        ("validation_before", before_validation),
        ("validation_after", after_validation),
        ("frozen_before", before_frozen),
        ("frozen_after", after_frozen),
    ]:
        if evaluation is not None:
            result.update({f"{prefix}_{key}": value for key, value in asdict(evaluation).items()})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.positive_weight <= 0 or args.unlabeled_weight <= 0:
        raise ValueError("Sample weights must be positive")
    if args.learning_rate <= 0 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Learning rate, epochs, and batch size must be positive")
    if not set(args.folds).issubset({1, 2, 3, 4, 5}):
        raise ValueError("--folds must be selected from 1 2 3 4 5")

    set_seed(args.seed)
    lab, seen_proteins = load_data(args)
    train_df, validation_df, frozen_df = split_by_protein(
        lab,
        seen_proteins,
        args.validation_fraction,
        args.seed,
        args.use_all_lab_proteins,
    )
    split_summary = {
        "adaptation_train": describe_split("adaptation_train", train_df),
        "adaptation_validation": describe_split("adaptation_validation", validation_df),
        "frozen_new_protein_positives": describe_split(
            "frozen_new_protein_positives", frozen_df
        ),
    }
    split_accounting = validate_split_accounting(
        lab,
        train_df,
        validation_df,
        frozen_df,
        args.use_all_lab_proteins,
    )
    train_positive_weight = (
        split_summary["adaptation_train"]["positives"] * args.positive_weight
    )
    train_unlabeled_weight = (
        split_summary["adaptation_train"]["unlabeled"] * args.unlabeled_weight
    )
    print(
        "adaptation_train_effective_weight: "
        f"positive={train_positive_weight:.1f}, "
        f"unlabeled={train_unlabeled_weight:.1f}"
    )

    model_dirs = {fold_number(path): path for path in args.model_dir.glob(E4_MODEL_PATTERN)}
    missing_folds = set(args.folds) - set(model_dirs)
    if missing_folds:
        raise FileNotFoundError(f"Missing source E4 folds: {sorted(missing_folds)}")
    if args.dry_run:
        print("Dry run complete; no model was loaded or written.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dirs[args.folds[0]])
    train_loader = build_loader(
        tokenizer,
        train_df,
        args.positive_weight,
        args.unlabeled_weight,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = build_loader(
        tokenizer,
        validation_df,
        args.positive_weight,
        args.unlabeled_weight,
        args.batch_size,
        shuffle=False,
    )
    frozen_loader = (
        build_loader(
            tokenizer,
            frozen_df,
            args.positive_weight,
            args.unlabeled_weight,
            args.batch_size,
            shuffle=False,
        )
        if len(frozen_df)
        else None
    )

    metrics = []
    for fold in args.folds:
        set_seed(args.seed + fold)
        metrics.append(
            train_one_fold(
                fold,
                model_dirs[fold],
                tokenizer,
                train_loader,
                validation_loader,
                frozen_loader,
                args,
                device,
            )
        )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.metrics_output, index=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "positive_weight": args.positive_weight,
        "unlabeled_weight": args.unlabeled_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_fraction": args.validation_fraction,
        "patience": args.patience,
        "threshold": args.threshold,
        "seed": args.seed,
        "folds": args.folds,
        "use_all_lab_proteins": args.use_all_lab_proteins,
        "splits": split_summary,
        "split_accounting": split_accounting,
        "warning": (
            "Adapted E4 scores are not compatible with the existing 5D MetaModel "
            "without retraining or recalibration."
        ),
    }
    manifest_path = args.output_dir / "adaptation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved fold metrics to {args.metrics_output}")
    print(f"Saved adaptation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
