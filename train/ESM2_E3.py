import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EsmModel,
    EsmPreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from group_splits import group_train_validation_split, load_fixed_group_folds


FILE_PATH = "./train_data_with_ids.xlsx"
MODEL_CACHE_DIR = "./modelscope"
LOCAL_MODEL_DIRS = [
    "./modelscope/esm2_t30_150M_UR50D",
    "./modelscope/models/facebook--esm2_t30_150M_UR50D/snapshots/master",
    "./facebook/esm2_t30_150M_UR50D",
]
MODELSCOPE_MODEL_ID = os.getenv("MODELSCOPE_MODEL_ID", "")
NUM_FOLDS = 5
NUM_LABELS = 2
VALIDATION_SIZE = 0.1
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 40
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 150
DECISION_THRESHOLD = 0.5
WINDOW_LENGTH = 51
CENTER_RESIDUE_INDEX = WINDOW_LENGTH // 2
# ESM adds <cls> before the 51 residue tokens, so residue index 25 is token index 26.
CENTER_TOKEN_INDEX = CENTER_RESIDUE_INDEX + 1
EXPERIMENT_NAME = "E3_cls_center_k"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def resolve_model_checkpoint() -> str:
    for local_model_dir in LOCAL_MODEL_DIRS:
        if os.path.isdir(local_model_dir):
            return local_model_dir

    if not MODELSCOPE_MODEL_ID:
        raise RuntimeError(
            "ModelScope model is not configured. Set MODELSCOPE_MODEL_ID to your ModelScope "
            "model name, or place the downloaded model under ./modelscope/esm2_t30_150M_UR50D/"
        )

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope is not installed. Run `.venv/bin/pip install modelscope` first."
        ) from exc

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return snapshot_download(model_id=MODELSCOPE_MODEL_ID, cache_dir=MODEL_CACHE_DIR)


def compute_binary_metrics(labels, probabilities, threshold: float = DECISION_THRESHOLD):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = recall_score(labels, predictions, zero_division=0)

    metrics = {
        "auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "mcc": matthews_corrcoef(labels, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "threshold": threshold,
    }
    return metrics


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = stable_softmax(logits)[:, 1]
    return compute_binary_metrics(labels, probabilities)


def build_dataset(tokenizer, sequences, labels):
    # Tokenize residues individually. Otherwise a run such as "*****" is
    # collapsed into one <unk> token by the ESM tokenizer.
    residue_tokens = [
        [tokenizer.pad_token if residue == "*" else residue for residue in sequence]
        for sequence in sequences
    ]
    tokenized = tokenizer(
        residue_tokens,
        is_split_into_words=True,
    )

    for sequence, input_ids, attention_mask in zip(
        sequences,
        tokenized["input_ids"],
        tokenized["attention_mask"],
    ):
        if len(input_ids) != len(sequence) + 2:
            raise ValueError("Expected ESM tokens: <cls> + 51 residues + <eos>")
        for token_index, token_id in enumerate(input_ids):
            if token_id == tokenizer.pad_token_id:
                attention_mask[token_index] = 0

    dataset = Dataset.from_dict(tokenized)
    return dataset.add_column("labels", list(labels))


class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(CustomClassificationHead, self).__init__()
        self.dense1 = nn.Linear(hidden_size, 512)
        self.ln = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(512, num_labels)

    def forward(self, pooled_features):
        x = pooled_features
        x = self.dense1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class ESM2CLSCenterKClassifier(EsmPreTrainedModel):
    """ESM2 classifier using concatenated CLS and center-K features."""

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

        cls_features = features[:, 0, :]
        center_k_features = features[:, CENTER_TOKEN_INDEX, :]
        combined_features = torch.cat([cls_features, center_k_features], dim=-1)
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


set_seed(SEED)

df = load_fixed_group_folds(FILE_PATH, n_splits=NUM_FOLDS, seed=SEED)
if not df["Sequence"].str.len().eq(WINDOW_LENGTH).all():
    raise ValueError(f"{EXPERIMENT_NAME} requires {WINDOW_LENGTH}-residue peptide windows")
if not df["Sequence"].str[CENTER_RESIDUE_INDEX].eq("K").all():
    raise ValueError(f"{EXPERIMENT_NAME} requires K at the center of every peptide window")
sequences = list(df["Sequence"])
labels = list(df["Label"])
sample_ids = list(df["Sample_ID"])
group_ids = list(df["Group_ID"])
fold_assignments = df["Fold"].to_numpy()

MODEL_CHECKPOINT = resolve_model_checkpoint()
print(f"Using model checkpoint: {MODEL_CHECKPOINT}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

all_predictions = []
all_labels = []
prediction_rows = []
fold_metrics_rows = []

for fold in range(1, NUM_FOLDS + 1):
    train_index = np.flatnonzero(fold_assignments != fold)
    test_index = np.flatnonzero(fold_assignments == fold)
    train_sequences_full = [sequences[i] for i in train_index]
    train_labels_full = [labels[i] for i in train_index]
    train_groups_full = [group_ids[i] for i in train_index]
    test_sequences = [sequences[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]

    inner_train_index, val_index = group_train_validation_split(
        train_labels_full,
        train_groups_full,
        validation_size=VALIDATION_SIZE,
        seed=SEED + fold,
    )
    train_sequences = [train_sequences_full[i] for i in inner_train_index]
    train_labels = [train_labels_full[i] for i in inner_train_index]
    val_sequences = [train_sequences_full[i] for i in val_index]
    val_labels = [train_labels_full[i] for i in val_index]

    train_dataset = build_dataset(tokenizer, train_sequences, train_labels)
    val_dataset = build_dataset(tokenizer, val_sequences, val_labels)
    test_dataset = build_dataset(tokenizer, test_sequences, test_labels)

    model = ESM2CLSCenterKClassifier.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.3,
        classifier_dropout=0.4,
    )

    run_name = (
        f"{EXPERIMENT_NAME}_fold_{fold}_{NUM_FOLDS}_grouped_lr{LEARNING_RATE}"
        f"_bs{BATCH_SIZE}_epochs{NUM_EPOCHS}"
    )
    output_dir = f"./trained_models/ESM2_ablation/{run_name}"

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        run_name=run_name,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=6,
                early_stopping_threshold=0.0001,
            )
        ],
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    fold_predictions = stable_softmax(predictions.predictions)[:, 1]
    fold_metrics = compute_binary_metrics(test_labels, fold_predictions)
    fold_metrics["fold"] = fold

    all_predictions.extend(fold_predictions)
    all_labels.extend(test_labels)
    prediction_rows.extend(
        {
            "Sample_ID": sample_ids[index],
            "Group_ID": group_ids[index],
            "Fold": fold,
            "label": int(labels[index]),
            "score": float(score),
        }
        for index, score in zip(test_index, fold_predictions)
    )
    fold_metrics_rows.append(fold_metrics)

    final_dir = (
        f"./trained_models/ESM2_ablation/{EXPERIMENT_NAME}_fold_{fold}_{NUM_FOLDS}"
        f"_grouped_lr{LEARNING_RATE}"
        f"_bs{BATCH_SIZE}_epochs{NUM_EPOCHS}_auc{fold_metrics['auc']:.6f}"
    )
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(
        f"Fold {fold} test metrics: "
        f"AUC={fold_metrics['auc']:.6f}, "
        f"PR-AUC={fold_metrics['pr_auc']:.6f}, "
        f"MCC={fold_metrics['mcc']:.6f}, "
        f"SN={fold_metrics['sensitivity']:.6f}, "
        f"SP={fold_metrics['specificity']:.6f}, "
        f"Precision={fold_metrics['precision']:.6f}, "
        f"F1={fold_metrics['f1']:.6f}, "
        f"ACC={fold_metrics['accuracy']:.6f}, "
        f"BalancedACC={fold_metrics['balanced_accuracy']:.6f}"
    )

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results_df = pd.DataFrame(prediction_rows)
score_prefix = f"ESM2_{EXPERIMENT_NAME}_grouped"
results_df.to_csv(f"./scores/{score_prefix}_y_label&score{NUM_FOLDS}.csv", index=False)

fold_metrics_df = pd.DataFrame(fold_metrics_rows)
fold_metrics_df = fold_metrics_df[
    [
        "fold",
        "auc",
        "pr_auc",
        "mcc",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "threshold",
    ]
]
fold_metrics_df.to_csv(f"./scores/{score_prefix}_fold_metrics{NUM_FOLDS}.csv", index=False)

overall_metrics = compute_binary_metrics(all_labels, all_predictions)
overall_metrics_df = pd.DataFrame([overall_metrics])
overall_metrics_df.to_csv(f"./scores/{score_prefix}_overall_metrics{NUM_FOLDS}.csv", index=False)

summary_series = fold_metrics_df.drop(columns=["fold"]).mean(numeric_only=True)
summary_df = pd.DataFrame(
    {
        "metric": summary_series.index,
        "mean": summary_series.values,
        "std": fold_metrics_df.drop(columns=["fold"]).std(numeric_only=True).values,
    }
)
summary_df.to_csv(
    f"./scores/{score_prefix}_fold_metrics_summary{NUM_FOLDS}.csv",
    index=False,
)

print(
    "Overall 5-fold test metrics: "
    f"AUC={overall_metrics['auc']:.6f}, "
    f"PR-AUC={overall_metrics['pr_auc']:.6f}, "
    f"MCC={overall_metrics['mcc']:.6f}, "
    f"SN={overall_metrics['sensitivity']:.6f}, "
    f"SP={overall_metrics['specificity']:.6f}, "
    f"Precision={overall_metrics['precision']:.6f}, "
    f"F1={overall_metrics['f1']:.6f}, "
    f"ACC={overall_metrics['accuracy']:.6f}, "
    f"BalancedACC={overall_metrics['balanced_accuracy']:.6f}"
)
