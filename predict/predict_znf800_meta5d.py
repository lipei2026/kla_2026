from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import predict as predict_module
from predict_znf800_e4 import (
    CENTER_RESIDUE_INDEX,
    E4_MODEL_DIR,
    E4_MODEL_PATTERN,
    ESM2CenterKMeanClassifier,
    WINDOW_LENGTH,
    fold_number,
    tokenize_windows,
)
from run_lab_external_test import kmors_encode, one_hot_binary_encode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "train" / "trained_models"
DEFAULT_INPUT = PROJECT_ROOT / "predict" / "znf800_candidates.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "predict" / "znf800_Meta5D_predictions.xlsx"
NUM_FOLDS = 5
THRESHOLD = 0.5
META_INPUT_ORDER = ["E4", "LSTM", "AAINDEX", "CKSAAP", "OBC"]
TOP10_PATH = PROJECT_ROOT / "train" / "top10.txt"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if "Sequence" not in df.columns or "Site" not in df.columns:
        raise ValueError("Input must contain Sequence and Site columns")
    result = df.copy()
    result["Sequence"] = result["Sequence"].astype(str).str.upper()
    if not result["Sequence"].str.len().eq(WINDOW_LENGTH).all():
        raise ValueError("Every candidate must use a 51-residue window")
    if not result["Sequence"].str[CENTER_RESIDUE_INDEX].eq("K").all():
        raise ValueError("Every candidate must have K at the window center")
    return result


def predict_e4_fold(model_dir, tokenized, device, batch_size):
    model = ESM2CenterKMeanClassifier.from_pretrained(model_dir).to(device)
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, tokenized["input_ids"].shape[0], batch_size):
            end = min(start + batch_size, tokenized["input_ids"].shape[0])
            logits = model(
                input_ids=tokenized["input_ids"][start:end].to(device),
                attention_mask=tokenized["attention_mask"][start:end].to(device),
            ).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=float)


def aaindex_encode(sequences):
    """Encode the 10x51 AAindex matrix without also calculating unused ACF."""
    with TOP10_PATH.open("r", encoding="utf-8") as handle:
        amino_acids = handle.readline().rstrip()[5:].split("\t")
        feature_rows = [line.rstrip().split("\t") for line in handle if line.strip()]
    residue_values = [
        {aa: float(value) for aa, value in zip(amino_acids, row[1:])}
        for row in feature_rows
    ]
    encoded = []
    for sequence in sequences:
        values = []
        for mapping in residue_values:
            values.extend(mapping.get(residue, 0.0) for residue in sequence)
        encoded.append(values)
    return encoded


def predict_lstm_fold(weight_path, encoded_sequences, device, batch_size):
    model = predict_module.LSTMModel(128, 256, 6, 2, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(encoded_sequences), batch_size):
            logits = model(encoded_sequences[start : start + batch_size].to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    del model
    return np.asarray(scores, dtype=float)


def predict_feature_fold(feature_name, weight_path, feature_tensor, device):
    config = predict_module.hidden_config[feature_name]
    model = predict_module.DNN(
        size=predict_module.size_dist[feature_name],
        hidden_sizes=config["hidden"],
        dropout_rate=config["dropout"],
        use_bn=config["bn"],
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        scores = model(feature_tensor.to(device)).cpu().numpy().reshape(-1)
    del model
    return scores.astype(float)


def predict_meta_fold(weight_path, base_scores, device):
    model = predict_module.hybridKla(len(META_INPUT_ORDER)).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    inputs = torch.tensor(base_scores, dtype=torch.float32, device=device)
    with torch.inference_mode():
        scores = model(inputs).cpu().numpy().reshape(-1)
    del model
    return scores.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-aligned five-dimensional MetaModel prediction for ZNF800."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw = pd.read_excel(input_path) if input_path.suffix.lower() == ".xlsx" else pd.read_csv(input_path)
    df = validate_candidates(raw)
    sequences = df["Sequence"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e4_dirs = {fold_number(path): path for path in E4_MODEL_DIR.glob(E4_MODEL_PATTERN)}
    if set(e4_dirs) != set(range(1, NUM_FOLDS + 1)):
        raise FileNotFoundError(f"Expected E4 folds 1-5, found {sorted(e4_dirs)}")
    tokenizer = predict_module.AutoTokenizer.from_pretrained(e4_dirs[1])
    tokenized = tokenize_windows(tokenizer, sequences)

    lstm_tensor = torch.tensor(predict_module.seq2num(sequences), dtype=torch.long)
    feature_tensors = {
        "AAINDEX": torch.tensor(np.asarray(aaindex_encode(sequences)), dtype=torch.float32),
        "CKSAAP": torch.tensor(np.asarray(kmors_encode(sequences, 1, mode="l")), dtype=torch.float32),
        "OBC": torch.tensor(np.asarray(one_hot_binary_encode(sequences, include_u=0)), dtype=torch.float32),
    }
    for name, tensor in feature_tensors.items():
        expected = predict_module.size_dist[name]
        if tensor.shape != (len(df), expected):
            raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match {(len(df), expected)}")

    print(
        f"Predicting {len(df)} candidate K sites on {device}; "
        f"fold-aligned input order: {META_INPUT_ORDER}"
    )
    meta_fold_scores = []
    base_scores_by_name = {name: [] for name in META_INPUT_ORDER}

    for fold in range(1, NUM_FOLDS + 1):
        print(f"Fold {fold}/{NUM_FOLDS}")
        fold_scores = {}
        fold_scores["E4"] = predict_e4_fold(e4_dirs[fold], tokenized, device, args.batch_size)
        lstm_path = require_file(
            TRAINED_MODELS_DIR / "LSTM" / f"fold_{fold}_5_E4folds_grouped_lstm.pth"
        )
        fold_scores["LSTM"] = predict_lstm_fold(
            lstm_path, lstm_tensor, device, args.batch_size
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

        stacked = np.column_stack([fold_scores[name] for name in META_INPUT_ORDER])
        meta_path = require_file(
            TRAINED_MODELS_DIR / f"meta_model_E4folds_grouped_5D_fold_{fold}_5.pth"
        )
        meta_scores = predict_meta_fold(meta_path, stacked, device)
        meta_fold_scores.append(meta_scores)
        df[f"Meta5D_fold{fold}_score"] = meta_scores
        for name in META_INPUT_ORDER:
            base_scores_by_name[name].append(fold_scores[name])
            df[f"{name}_fold{fold}_score"] = fold_scores[name]

    meta_matrix = np.vstack(meta_fold_scores)
    df["Meta5D_score"] = meta_matrix.mean(axis=0)
    df["Meta5D_score_std"] = meta_matrix.std(axis=0, ddof=1)
    df["Meta5D_prediction"] = (df["Meta5D_score"] >= THRESHOLD).astype(int)
    for name, fold_scores in base_scores_by_name.items():
        df[f"{name}_score"] = np.vstack(fold_scores).mean(axis=0)

    df = df.sort_values("Meta5D_score", ascending=False).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    positive = df[df["Meta5D_prediction"] == 1]
    summary_path = output_path.with_suffix(".summary.txt")
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Fold-aligned five-dimensional MetaModel prediction\n")
        handle.write(f"input_order: {', '.join(META_INPUT_ORDER)}\n")
        handle.write(f"threshold: {THRESHOLD:.2f}\n")
        handle.write(f"candidate_K_sites: {len(df)}\n")
        handle.write(f"predicted_positive_sites: {len(positive)}\n")
        handle.write(f"mean_Meta5D_score: {df['Meta5D_score'].mean():.6f}\n")
        handle.write(f"max_Meta5D_score: {df['Meta5D_score'].max():.6f}\n")
        handle.write(f"hit_rate@0.5: {(df['Meta5D_score'] >= 0.5).mean():.6f}\n")
        handle.write(f"hit_rate@0.7: {(df['Meta5D_score'] >= 0.7).mean():.6f}\n")
        handle.write(f"hit_rate@0.9: {(df['Meta5D_score'] >= 0.9).mean():.6f}\n")
        if "Label" in df.columns and set(df["Label"].dropna().unique()) == {1}:
            handle.write(
                "note: all labels are positive; hit_rate equals sensitivity/recall, "
                "and AUC/specificity/MCC are undefined\n"
            )
        if len(positive):
            sites = ", ".join(f"K{int(site)}" for site in positive["Site"])
            handle.write(f"positive_sites_ranked: {sites}\n")

    print(f"Saved predictions to {output_path}")
    print(f"Saved summary to {summary_path}")
    print(
        df[["Site", "Meta5D_score", "Meta5D_score_std", "Meta5D_prediction"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
