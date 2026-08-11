from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, EsmModel, EsmPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "predict" / "znf800_candidates.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "predict" / "znf800_E4_predictions.xlsx"
E4_MODEL_DIR = PROJECT_ROOT / "train" / "trained_models" / "ESM2_ablation"
E4_MODEL_PATTERN = "E4_center_k_mean_fold_*_5_grouped_lr5e-05_bs32_epochs40_auc*"
WINDOW_LENGTH = 51
CENTER_RESIDUE_INDEX = WINDOW_LENGTH // 2
CENTER_TOKEN_INDEX = CENTER_RESIDUE_INDEX + 1
THRESHOLD = 0.5


class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, 512)
        self.ln = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(512, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dense1(features)
        features = self.ln(features)
        features = self.relu(features)
        features = self.dropout(features)
        return self.dense2(features)


class ESM2CenterKMeanClassifier(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esm = EsmModel(config, add_pooling_layer=False)
        self.classifier = CustomClassificationHead(config.hidden_size * 2, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        features = outputs.last_hidden_state
        valid_mask = attention_mask.bool()
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False
        valid_mask &= input_ids.ne(self.config.pad_token_id)
        mask = valid_mask.unsqueeze(-1).to(features.dtype)
        mean_features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        center_k_features = features[:, CENTER_TOKEN_INDEX, :]
        combined_features = torch.cat([center_k_features, mean_features], dim=-1)
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def tokenize_windows(tokenizer, sequences: list[str]) -> dict[str, torch.Tensor]:
    residue_tokens = [
        [tokenizer.pad_token if residue == "*" else residue for residue in sequence]
        for sequence in sequences
    ]
    tokenized = tokenizer(
        residue_tokens,
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
    )
    tokenized["attention_mask"] = tokenized["attention_mask"].masked_fill(
        tokenized["input_ids"].eq(tokenizer.pad_token_id),
        0,
    )
    if tokenized["input_ids"].shape[1] != WINDOW_LENGTH + 2:
        raise ValueError("Expected <cls> + 51 residue tokens + <eos>")
    center_ids = tokenized["input_ids"][:, CENTER_TOKEN_INDEX]
    k_token_id = tokenizer.convert_tokens_to_ids("K")
    if not center_ids.eq(k_token_id).all():
        raise ValueError("Tokenized center position is not K for every candidate")
    return tokenized


def fold_number(model_dir: Path) -> int:
    return int(model_dir.name.split("_fold_", 1)[1].split("_", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ZNF800 Kla sites with the E4 ESM2 ensemble.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-dir", default=str(E4_MODEL_DIR))
    parser.add_argument("--model-pattern", default=E4_MODEL_PATTERN)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_excel(input_path) if input_path.suffix.lower() == ".xlsx" else pd.read_csv(input_path)
    if "Sequence" not in df.columns or "Site" not in df.columns:
        raise ValueError("Input must contain Sequence and Site columns")

    df = df.copy()
    df["Sequence"] = df["Sequence"].astype(str).str.upper()
    if not df["Sequence"].str.len().eq(WINDOW_LENGTH).all():
        raise ValueError("Every ZNF800 candidate must use a 51-residue window")
    if not df["Sequence"].str[CENTER_RESIDUE_INDEX].eq("K").all():
        raise ValueError("Every ZNF800 candidate must have K at the window center")

    model_dirs = sorted(
        Path(args.model_dir).glob(args.model_pattern),
        key=fold_number,
    )
    if len(model_dirs) != 5:
        raise FileNotFoundError(f"Expected 5 final E4 fold models, found {len(model_dirs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predicting {len(df)} ZNF800 K sites on {device} with {len(model_dirs)} E4 folds")
    tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
    tokenized = tokenize_windows(tokenizer, df["Sequence"].tolist())
    fold_scores = []

    for model_dir in model_dirs:
        fold = fold_number(model_dir)
        print(f"Loading E4 fold {fold}: {model_dir.name}")
        model = ESM2CenterKMeanClassifier.from_pretrained(model_dir).to(device)
        model.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(df), args.batch_size):
                end = min(start + args.batch_size, len(df))
                batch = {
                    key: value[start:end].to(device)
                    for key, value in tokenized.items()
                    if key in {"input_ids", "attention_mask"}
                }
                logits = model(**batch).logits
                scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        scores = np.asarray(scores, dtype=float)
        df[f"E4_fold{fold}_score"] = scores
        fold_scores.append(scores)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_matrix = np.vstack(fold_scores)
    df["E4_score"] = score_matrix.mean(axis=0)
    df["E4_score_std"] = score_matrix.std(axis=0, ddof=1)
    df["E4_prediction"] = (df["E4_score"] >= THRESHOLD).astype(int)
    df = df.sort_values("E4_score", ascending=False).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.txt")
    positive = df[df["E4_prediction"] == 1]
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("ZNF800 prediction with five-fold E4 ESM2 ensemble\n")
        handle.write(f"threshold: {THRESHOLD:.2f}\n")
        handle.write(f"candidate_K_sites: {len(df)}\n")
        handle.write(f"predicted_positive_sites: {len(positive)}\n")
        handle.write(f"mean_E4_score: {df['E4_score'].mean():.6f}\n")
        handle.write(f"max_E4_score: {df['E4_score'].max():.6f}\n")
        if len(positive):
            sites = ", ".join(f"K{int(site)}" for site in positive["Site"])
            handle.write(f"positive_sites_ranked: {sites}\n")

    print(f"Saved predictions to {output_path}")
    print(f"Saved summary to {summary_path}")
    print(df[["Site", "E4_score", "E4_score_std", "E4_prediction"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
