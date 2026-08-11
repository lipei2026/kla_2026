# GitHub upload notes

This repository package contains the source code, public benchmark input,
metric summaries, lightweight prediction examples, figures, and experiment
report. It intentionally does not contain the following local artifacts:

- the existing `.git/` directory (its historical objects are about 107 GB);
- `.venv/` and Python/tool caches;
- `train/trained_models/` model and optimizer checkpoints;
- `train/modelscope/` downloaded ESM2 files;
- `train/five_features/` generated feature matrices;
- UniProt download caches;
- private laboratory spreadsheets and row-level laboratory predictions;
- full out-of-fold score tables (metric summaries are retained).

The omitted artifacts are not required in Git history. Dependencies can be
restored with `pip install -r requirements.txt`; features and checkpoints can
be regenerated with the scripts in `train/`. If trained weights need to be
distributed, publish inference-only checkpoints separately through a model
repository or a versioned data archive and add their download links to the
README. Do not publish optimizer states or private laboratory data.

## Create and publish a fresh repository

Extract the release package, then run:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Starting with a fresh repository is important: pushing the old local `.git`
history would also try to upload its oversized model objects.

Before making the repository public, choose a license and add a `LICENSE`
file. Also review the public benchmark dataset's original redistribution terms
and add its citation/source information to the README.
