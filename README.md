# Kla 2026: Protein lysine lactylation site prediction

## Project overview

This repository implements a machine-learning pipeline for predicting lysine
lactylation (Kla) sites from fixed-length peptide windows centred on lysine.
For every candidate site, the model outputs the probability that the centre
lysine is lactylated.

The project focuses on three practical questions: how to extract useful
representations from a pretrained protein Transformer, how to evaluate
site-level predictions without leaking proteins across folds, and whether
different sequence and statistical models provide complementary information.

The implemented pipeline combines ESM2 representations, an LSTM sequence
model, and five handcrafted descriptors (ACF, AAINDEX, CKSAAP, OBC, and
PSEAAC). A stacking classifier integrates their out-of-fold predictions.
Evaluation uses five-fold `StratifiedGroupKFold`, ensuring that sites from the
same protein remain in the same fold. ESM2 pooling ablations, model comparison,
and domain-adaptation experiments are documented in
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md).

## Main components

### Prediction (`predict/`)

- `predict.py`: batch prediction entry point
- `predict_znf800_*.py`: ZNF800 case-study inference
- `predict_o15156_meta5d.py`: O15156/ZBTB7B case-study inference
- `plot_znf800_lollipop.py`: site-level lollipop visualization

### Training (`train/`)

- `ESM2.py`: CLS baseline
- `ESM2_E3.py` and `ESM2_E4.py`: ESM2 representation ablations
- `LSTM.py`: LSTM baseline
- `Encoding_fivefeatures.py` and `Five_feature_train.py`: handcrafted features
- `Meta_model_5D.py`: five-input stacking model
- `finetune_lab_e4_weighted.py`: weighted domain adaptation of the E4 head

### Protein-aware cross-validation

Protein identifiers can be reconstructed by matching the benchmark peptide
windows against its supplementary table and UniProt sequences:

```bash
.venv/bin/python data/recover_protein_ids.py --download-missing
```

This creates `train/train_data_with_ids.xlsx`. Exact ambiguous matches are
retained in a common `Group_ID`; unresolved samples are assigned `Fold=0` and
excluded from training. The training scripts use the shared protein-level
`StratifiedGroupKFold` assignments in this file. Run training scripts from the
`train/` directory so their relative input and output paths resolve correctly.


## Installation

### Using Conda (Recommended)

```bash
conda create -n hybridkla python=3.10
conda activate hybridkla
pip install -r requirements.txt
```

### Using Pip Only

```bash
pip install -r requirements.txt
```

## Repository artifacts

Large model checkpoints, downloaded ESM2 files, generated feature matrices,
and private laboratory datasets are deliberately excluded from Git. See
[`GITHUB_UPLOAD.md`](GITHUB_UPLOAD.md) for the artifact policy and instructions
for publishing this project as a fresh GitHub repository.
