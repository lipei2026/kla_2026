from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICT_DIR = PROJECT_ROOT / "predict"
TRAIN_DIR = PROJECT_ROOT / "train"
TRAINED_MODELS_DIR = TRAIN_DIR / "trained_models"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "lab_test_data.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "predict" / "lab_external_test_predictions.xlsx"

sys.path.insert(0, str(PREDICT_DIR))

import predict as predict_module  # noqa: E402


TOP10_PATH = TRAIN_DIR / "top10.txt"


def load_input_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input file must be .xlsx, .xls or .csv")


def save_output_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError("Output file must be .xlsx, .xls or .csv")


def acf_encode(sequences: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    data = DataFrame(sequences)
    feature_values = {}
    feature_names = []

    with TOP10_PATH.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().rstrip()[5:]
        amino_acids = first_line.split("\t")
        for line in handle:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split("\t")
            feature_values[parts[0]] = parts[1:]
            feature_names.append(parts[0])

    def aaindex_flatten(sequence: str) -> list[float]:
        values: list[float] = []
        sequence = sequence.replace("*", "0").replace("X", "0").replace("B", "0").replace("U", "0")
        for feature_name in feature_names:
            residues = list(sequence)
            for idx, amino_acid in enumerate(amino_acids):
                residues = [feature_values[feature_name][idx] if aa == amino_acid else aa for aa in residues]
            values.extend(float(item) for item in residues)
        return values

    def acf_single(sequence: str) -> list[float]:
        aaindex_rows = []
        sequence = sequence.replace("*", "0").replace("X", "0").replace("B", "0").replace("U", "0")
        for feature_name in feature_names:
            residues = list(sequence)
            for idx, amino_acid in enumerate(amino_acids):
                residues = [feature_values[feature_name][idx] if aa == amino_acid else aa for aa in residues]
            aaindex_rows.append([float(item) for item in residues])

        encoded = np.zeros((len(aaindex_rows), len(sequence)))
        for row_index, seq_values in enumerate(aaindex_rows):
            for lag in range(len(sequence)):
                values = [seq_values[i] * seq_values[i + lag] for i in range(len(seq_values) - lag)]
                encoded[row_index, lag] = round(sum(values) / len(values), 2)
        return encoded.flatten().tolist()

    data["acf"] = data[0].apply(acf_single)
    data["aaindex"] = data[0].apply(aaindex_flatten)
    return np.array(list(data["acf"]), dtype=np.float32).tolist(), np.array(list(data["aaindex"]), dtype=np.float32).tolist()


def kmors_encode(sequences: list[str], km: int = 1, mode: str = "l") -> list[list[int]]:
    data = DataFrame(sequences)
    pairs = []
    if mode == "d":
        alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    else:
        alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y"]
    for aa in alphabet:
        for bb in alphabet:
            pairs.append(aa + bb)

    def encode(sequence: str) -> list[int]:
        values = []
        for gap in range(km):
            observed = []
            for index, aa in enumerate(sequence):
                if index + gap + 1 < len(sequence):
                    observed.append(aa + sequence[index + gap + 1])
            values.extend(float(observed.count(pair)) for pair in pairs)
        return values

    data["encoded"] = data[0].apply(encode)
    return np.array(list(data["encoded"]), dtype=np.float32).tolist()


def aac_encode(sequences: list[str]) -> list[list[float]]:
    data = DataFrame(sequences)
    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    seq_len = len(sequences[0])

    def encode(sequence: str) -> list[float]:
        return [float(sequence.count(aa) / seq_len) for aa in amino_acids]

    data["encoded"] = data[0].apply(encode)
    return np.array(list(data["encoded"]), dtype=np.float32).tolist()


def one_hot_binary_encode(sequences: list[str], include_u: int = 0) -> list[list[float]]:
    data = DataFrame(sequences)
    alphabet = Series(
        range(0, 21),
        index=["K", "L", "A", "E", "V", "G", "S", "D", "I", "T", "R", "*", "P", "Q", "N", "F", "Y", "M", "H", "C", "W"],
    )
    if include_u == 0:
        alphabet = Series(
            range(0, 22),
            index=["K", "L", "A", "E", "V", "G", "S", "D", "I", "T", "R", "*", "P", "Q", "N", "F", "Y", "M", "H", "C", "W", "U"],
        )
    alphabet[:] = range(len(alphabet))
    valid_residues = set(alphabet.index)

    def encode(sequence: str) -> list[int]:
        sequence = sequence.replace("X", "*").replace("B", "*")
        filtered = [aa for aa in sequence if aa in valid_residues]
        return list(alphabet[filtered])

    data["encoded"] = data[0].apply(encode)
    values = np.array(list(data["encoded"]), dtype=np.int64)

    def to_one_hot(row: np.ndarray) -> np.ndarray:
        return np.eye(len(alphabet))[row].flatten()

    encoded = np.array(list(map(to_one_hot, values)), dtype=np.float32)
    return encoded.tolist()


def load_lstm_ensemble(device: torch.device) -> list[torch.nn.Module]:
    model_paths = sorted((TRAINED_MODELS_DIR / "LSTM").glob("fold_*_5_lstm.pth"))
    models = []
    for model_path in model_paths:
        model = predict_module.LSTMModel(128, 256, 6, 2, 0.2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError("No LSTM fold models found in train/trained_models/LSTM")
    return models


def load_esm2_ensemble(device: torch.device) -> tuple[AutoTokenizer, list[torch.nn.Module]]:
    model_dirs = sorted(TRAINED_MODELS_DIR.glob("ESM2/fold_*_5_lr5e-05_bs32_epochs40_auc*"))
    if not model_dirs:
        raise FileNotFoundError("No ESM2 fold models found in train/trained_models/ESM2")

    tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
    models = []
    for model_dir in model_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=2,
            hidden_dropout_prob=0.3,
            classifier_dropout=0.4,
        )
        hidden_size = model.config.hidden_size
        model.classifier = predict_module.CustomClassificationHead(hidden_size, 2)
        weights = load_file(str(model_dir / "model.safetensors"))
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        models.append(model)
    return tokenizer, models


def load_feature_ensemble(device: torch.device) -> dict[str, list[torch.nn.Module]]:
    models: dict[str, list[torch.nn.Module]] = {}
    for feature_name, config in predict_module.hidden_config.items():
        feature_paths = sorted((TRAINED_MODELS_DIR / feature_name).glob(f"{feature_name}*_5_DNN_.pth"))
        feature_models = []
        for feature_path in feature_paths:
            model = predict_module.DNN(
                size=predict_module.size_dist[feature_name],
                hidden_sizes=config["hidden"],
                dropout_rate=config["dropout"],
                use_bn=config["bn"],
            )
            model.load_state_dict(torch.load(feature_path, map_location=device))
            model.to(device)
            model.eval()
            feature_models.append(model)
        if not feature_models:
            raise FileNotFoundError(f"No fold models found for feature {feature_name}")
        models[feature_name] = feature_models
    return models


def load_meta_ensemble(device: torch.device, input_size: int = 7) -> list[torch.nn.Module]:
    model_paths = sorted(TRAINED_MODELS_DIR.glob("meta_model_*_5.pth"))
    models = []
    for model_path in model_paths:
        model = predict_module.hybridKla(input_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError("No meta_model fold weights found in train/trained_models")
    return models


def average_lstm_scores(models: list[torch.nn.Module], sequences: list[str], device: torch.device) -> np.ndarray:
    encoded = torch.tensor(predict_module.seq2num(sequences), dtype=torch.long, device=device)
    scores = []
    with torch.no_grad():
        for model in models:
            logits = model(encoded)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.mean(np.vstack(scores), axis=0)


def average_esm2_scores(
    tokenizer: AutoTokenizer,
    models: list[torch.nn.Module],
    sequences: list[str],
    device: torch.device,
) -> np.ndarray:
    tokenized = tokenizer(sequences, padding=True, truncation=True, max_length=51, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    scores = []
    with torch.no_grad():
        for model in models:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            scores.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.mean(np.vstack(scores), axis=0)


def extract_feature_arrays(sequences: list[str]) -> dict[str, np.ndarray]:
    acf_values, aaindex_values = acf_encode(sequences)
    obc_values = one_hot_binary_encode(sequences, include_u=0)
    cksaap_values = kmors_encode(sequences, 1, mode="l")
    pseaac_values = aac_encode(sequences)
    return {
        "ACF": np.asarray(acf_values, dtype=np.float32),
        "AAINDEX": np.asarray(aaindex_values, dtype=np.float32),
        "OBC": np.asarray(obc_values, dtype=np.float32),
        "CKSAAP": np.asarray(cksaap_values, dtype=np.float32),
        "PSEAAC": np.asarray(pseaac_values, dtype=np.float32),
    }


def average_feature_scores(
    models: dict[str, list[torch.nn.Module]],
    feature_arrays: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for feature_name, feature_models in models.items():
            feature_tensor = torch.tensor(feature_arrays[feature_name], dtype=torch.float32, device=device)
            fold_scores = [model(feature_tensor).cpu().numpy().reshape(-1) for model in feature_models]
            outputs[feature_name] = np.mean(np.vstack(fold_scores), axis=0)
    return outputs


def average_meta_scores(models: list[torch.nn.Module], stacked_features: np.ndarray, device: torch.device) -> np.ndarray:
    meta_input = torch.tensor(stacked_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        fold_scores = [model(meta_input).cpu().numpy().reshape(-1) for model in models]
    return np.mean(np.vstack(fold_scores), axis=0)


def summarize_positive_only(scores: np.ndarray) -> dict[str, float]:
    return {
        "count": float(len(scores)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "hit_rate@0.5": float(np.mean(scores >= 0.5)),
        "hit_rate@0.7": float(np.mean(scores >= 0.7)),
        "hit_rate@0.9": float(np.mean(scores >= 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external prediction on standardized laboratory test data.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to standardized lab test file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to prediction output file.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ESM2/LSTM prediction.")
    args = parser.parse_args()

    input_df = load_input_table(Path(args.input))
    if "Sequence" not in input_df.columns:
        raise ValueError("Input file must contain a 'Sequence' column.")

    input_df = input_df.copy()
    input_df["Sequence"] = input_df["Sequence"].astype(str).str.upper()
    sequences = input_df["Sequence"].tolist()

    device = torch.device("cpu")
    print(f"Loaded {len(sequences)} lab sequences from {args.input}")

    print("Loading model ensembles...")
    lstm_models = load_lstm_ensemble(device)
    tokenizer, esm2_models = load_esm2_ensemble(device)
    feature_models = load_feature_ensemble(device)
    meta_models = load_meta_ensemble(device, input_size=7)

    all_results = []
    total = len(sequences)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_df = input_df.iloc[start:end].copy()
        batch_sequences = batch_df["Sequence"].tolist()

        feature_arrays = extract_feature_arrays(batch_sequences)
        lstm_scores = average_lstm_scores(lstm_models, batch_sequences, device)
        esm2_scores = average_esm2_scores(tokenizer, esm2_models, batch_sequences, device)
        feature_scores = average_feature_scores(feature_models, feature_arrays, device)

        stacked_features = np.column_stack(
            [
                esm2_scores,
                lstm_scores,
                feature_scores["ACF"],
                feature_scores["AAINDEX"],
                feature_scores["CKSAAP"],
                feature_scores["OBC"],
                feature_scores["PSEAAC"],
            ]
        )
        meta_scores = average_meta_scores(meta_models, stacked_features, device)

        batch_df["ESM2_score"] = esm2_scores
        batch_df["LSTM_score"] = lstm_scores
        batch_df["ACF_score"] = feature_scores["ACF"]
        batch_df["AAINDEX_score"] = feature_scores["AAINDEX"]
        batch_df["CKSAAP_score"] = feature_scores["CKSAAP"]
        batch_df["OBC_score"] = feature_scores["OBC"]
        batch_df["PSEAAC_score"] = feature_scores["PSEAAC"]
        batch_df["MetaModel_score"] = meta_scores
        all_results.append(batch_df)
        print(f"Processed {end}/{total}")

    result_df = pd.concat(all_results, ignore_index=True)
    result_df = result_df.sort_values("MetaModel_score", ascending=False).reset_index(drop=True)
    save_output_table(result_df, Path(args.output))

    summary = summarize_positive_only(result_df["MetaModel_score"].to_numpy())
    summary_path = Path(args.output).with_suffix(".summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Positive-only external test summary\n")
        handle.write("Note: raw_data_lab.xlsx contains positive sites only, so AUC is not reported here.\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value:.6f}\n")

    print(f"Saved predictions to {args.output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
