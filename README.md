# HybridKla: A hybrid deep learning framework for lactylation site prediction

## 🧬 Introduction

Lysine lactylation (Kla), a novel lactate-derived post-translational modification (PTM), is involved in a myriad of biological processes and complex diseases.

Here, we report a friendly online service.We manually collected 23,984 Kla sites across 7,297 proteins from 14 species to construct the comprehensive Kla benchmark dataset.

We designed a multi-feature hybrid system that combines ESM2 protein-language-model representations, an LSTM sequence model, and five handcrafted descriptors (ACF, AAINDEX, CKSAAP, OBC, and PSEAAC). The current protein-aware evaluation uses five-fold `StratifiedGroupKFold` splitting to keep peptides from the same protein in one fold. ESM2 representation ablations and the final stacking experiments are described in `EXPERIMENT_REPORT.md`.
![Model Architecture](model.jpg)

## 🔧 Key Components

### Prediction Module (`predict/`)
- **predict.py**: Main script for batch processing protein sequences
- **models/**: Contains trained models for prediction
  - **ESM2/**: Fine-tuned protein language model files
  - **LSTM.pth**: Sequence pattern recognition model
  - **Meta-model checkpoints**: Final stacking classifiers
  - **Feature-specific models**: ACF, AAINDEX, OBC, CKSAAP, and PSEAAC classifiers

### Training Module (`train/`)
- **ESM2.py**: ESM2 language model fine-tuning script
- **LSTM.py**: LSTM model training implementation
- **Five_feature_train.py**: Training script for five feature-specific models
- **Encoding_fivefeatures.py**: Feature encoding implementation
- **Meta_model_5D.py**: Five-dimensional meta-model training

### Protein-aware cross-validation

The anonymized peptide identifiers can be recovered from the public HybridKla
Supplementary Table S1 and UniProt:

```bash
.venv/bin/python data/recover_protein_ids.py --download-missing
```

This creates `train/train_data_with_ids.xlsx`. Exact ambiguous matches are
retained in a common `Group_ID`; unresolved samples are assigned `Fold=0` and
excluded from training. The training scripts use the shared protein-level
`StratifiedGroupKFold` assignments in this file. Run training scripts from the
`train/` directory so their relative input and output paths resolve correctly.


## 🚀 Installation

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
