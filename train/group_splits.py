"""Shared leakage-safe protein-group cross-validation helpers."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_SPLIT_COLUMNS = {"Label", "Group_ID"}
FIXED_FOLD_COLUMNS = {"Sample_ID", "Label", "Group_ID", "Fold"}


def resolved_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Drop samples whose source protein could not be recovered."""
    result = df.copy()
    if "Match_Status" in result.columns:
        unresolved = result["Match_Status"].eq("unresolved")
        if unresolved.any():
            print(f"Excluding {int(unresolved.sum())} unresolved protein-ID samples")
        result = result.loc[~unresolved].copy()
    return result.reset_index(drop=True)


def assign_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    missing = REQUIRED_SPLIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing group-split columns: {sorted(missing)}")

    result = resolved_samples(df)
    use_existing = "Fold" in result.columns and set(result["Fold"].unique()) == set(
        range(1, n_splits + 1)
    )
    if not use_existing:
        result["Fold"] = 0
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold, (_, test_index) in enumerate(
            splitter.split(result.index, result["Label"], result["Group_ID"]),
            start=1,
        ):
            result.loc[test_index, "Fold"] = fold

    group_fold_counts = result.groupby("Group_ID")["Fold"].nunique()
    if (group_fold_counts > 1).any():
        leaked = group_fold_counts[group_fold_counts > 1].index[:5].tolist()
        raise RuntimeError(f"Group leakage detected for groups: {leaked}")
    return result


def load_fixed_group_folds(
    file_path: str,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Load and validate the persisted protein folds used by the ESM2 E4 run.

    Unlike ``assign_group_folds``, this function deliberately refuses to create
    new folds. Downstream base models must use the exact same outer test folds
    as E4, rather than merely generating another valid group split.
    """
    result = resolved_samples(pd.read_excel(file_path))
    missing = FIXED_FOLD_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"Missing fixed-fold columns: {sorted(missing)}")
    expected_folds = set(range(1, n_splits + 1))
    observed_folds = set(result["Fold"].unique())
    if observed_folds != expected_folds:
        raise ValueError(
            f"Expected persisted E4 folds {sorted(expected_folds)}, "
            f"found {sorted(observed_folds)}"
        )
    if result["Sample_ID"].duplicated().any():
        duplicates = result.loc[result["Sample_ID"].duplicated(), "Sample_ID"].head().tolist()
        raise ValueError(f"Duplicate Sample_ID values in fold reference: {duplicates}")

    # Reuse the central leakage checks without allowing fold regeneration.
    result = assign_group_folds(result, n_splits=n_splits, seed=seed)
    print(
        f"Using persisted E4 protein folds: {len(result)} samples, "
        f"{result['Group_ID'].nunique()} groups"
    )
    return result


def iter_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    assigned = assign_group_folds(df, n_splits=n_splits, seed=seed)
    for fold in range(1, n_splits + 1):
        test_index = np.flatnonzero(assigned["Fold"].to_numpy() == fold)
        train_index = np.flatnonzero(assigned["Fold"].to_numpy() != fold)
        yield fold, train_index, test_index


def group_train_validation_split(
    labels,
    groups,
    validation_size: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one stratified group-aware train/validation split."""
    n_splits = max(2, round(1.0 / validation_size))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    train_index, validation_index = next(
        splitter.split(np.arange(len(labels)), labels, groups)
    )
    return np.asarray(train_index), np.asarray(validation_index)
