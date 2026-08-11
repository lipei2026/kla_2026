"""Predict O15156 Kla sites with lab-adapted E4 and draw a lollipop plot.

The embedded sequence is the 539-aa UniProt canonical isoform O15156-1,
sequence version 2. Its checksum is verified before candidate windows are
created. Prediction uses lab-weighted E4 and its independently retrained,
fold-aligned 5D MetaModel.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICT_DIR = PROJECT_ROOT / "predict"
FIGURES_DIR = PROJECT_ROOT / "figures"
UNIPROT_ID = "O15156"
GENE_NAME = "ZBTB7B"
PROTEIN_NAME = "Zinc finger and BTB domain-containing protein 7B"
UNIPROT_URL = "https://www.uniprot.org/uniprotkb/O15156/entry"
EXPECTED_LENGTH = 539
EXPECTED_MD5 = "74E8844480192C0EDC502E4820D61E8F"
WINDOW_LENGTH = 51
WINDOW_RADIUS = WINDOW_LENGTH // 2

CANONICAL_SEQUENCE = (
    "MGSPEDDLIGIPFPDHSSELLSCLNEQRQLGHLCDLTIRTQGLEYRTHRAVLAACSHYFK"
    "KLFTEGGGGAVMGAGGSGTATGGAGAGVCELDFVGPEALGALLEFAYTATLTTSSANMPA"
    "VLQAARLLEIPCVIAACMEILQGSGLEAPSPDEDDCERARQYLEAFATATASGVPNGEDS"
    "PPQVPLPPPPPPPPRPVARRSRKPRKAFLQTKGARANHLVPEVPTVPAHPLTYEEEEVAG"
    "RVGSSGGSGPGDSYSPPTGTASPPEGPQSYEPYEGEEEEEELVYPPAYGLAQGGGPPLSP"
    "EELGSDEDAIDPDLMAYLSSLHQDNLAPGLDSQDKLVRKRRSQMPQECPVCHKIIHGAGK"
    "LPRHMRTHTGEKPFACEVCGVRFTRNDKLKIHMRKHTGERPYSCPHCPARFLHSYDLKNH"
    "MHLHTGDRPYECHLCHKAFAKEDHLQRHLKGQNCLEVRTRRRRKDDAPPHYPPPSTAAAS"
    "PAGLDLSNGHLDTFRLSLARFWEQSAPTGPPVSTPGPPDDDEEEGAPTTPQAEGAMESS"
)


def validate_sequence(sequence: str) -> None:
    if len(sequence) != EXPECTED_LENGTH:
        raise ValueError(f"Expected {EXPECTED_LENGTH} residues, found {len(sequence)}")
    checksum = hashlib.md5(sequence.encode("ascii")).hexdigest().upper()
    if checksum != EXPECTED_MD5:
        raise ValueError(f"UniProt sequence checksum mismatch: {checksum}")


def centered_window(sequence: str, center_index: int) -> str:
    residues = []
    for index in range(center_index - WINDOW_RADIUS, center_index + WINDOW_RADIUS + 1):
        residues.append(sequence[index] if 0 <= index < len(sequence) else "*")
    return "".join(residues)


def build_candidates(sequence: str) -> pd.DataFrame:
    rows = []
    for index, residue in enumerate(sequence):
        if residue != "K":
            continue
        rows.append(
            {
                "UniProt_ID": UNIPROT_ID,
                "Gene": GENE_NAME,
                "Protein_Name": PROTEIN_NAME,
                "Protein_Length": len(sequence),
                "Site": index + 1,
                "Sequence": centered_window(sequence, index),
                "Source": UNIPROT_URL,
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No lysine candidates found")
    if not candidates["Sequence"].str.len().eq(WINDOW_LENGTH).all():
        raise ValueError("Candidate window length validation failed")
    if not candidates["Sequence"].str[WINDOW_RADIUS].eq("K").all():
        raise ValueError("Candidate center-residue validation failed")
    return candidates


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PREDICT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict O15156/ZBTB7B Kla sites with lab-weighted E4 and the new 5D MetaModel."
        )
    )
    parser.add_argument(
        "--candidates",
        default=str(PREDICT_DIR / "o15156_candidates.xlsx"),
    )
    parser.add_argument(
        "--output",
        default=str(PREDICT_DIR / "o15156_LabWeightedE4_Meta5D_predictions.xlsx"),
    )
    parser.add_argument(
        "--figure",
        default=str(FIGURES_DIR / "O15156_ZBTB7B_lollipop_Meta5D.png"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=9)
    args = parser.parse_args()

    validate_sequence(CANONICAL_SEQUENCE)
    candidates = build_candidates(CANONICAL_SEQUENCE)
    candidates_path = Path(args.candidates).resolve()
    output_path = Path(args.output).resolve()
    figure_path = Path(args.figure).resolve()
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_excel(candidates_path, index=False)
    print(f"Saved {len(candidates)} candidate K sites to {candidates_path}", flush=True)

    run_command(
        [
            sys.executable,
            str(PREDICT_DIR / "predict_znf800_lab_weighted_meta5d.py"),
            "--input",
            str(candidates_path),
            "--output",
            str(output_path),
            "--batch-size",
            str(args.batch_size),
            "--threshold",
            str(args.threshold),
            "--protein-name",
            "O15156/ZBTB7B",
        ]
    )
    run_command(
        [
            sys.executable,
            str(PREDICT_DIR / "plot_znf800_lollipop.py"),
            "--input",
            str(output_path),
            "--output",
            str(figure_path),
            "--score-column",
            "LabWeighted_Meta5D_score",
            "--threshold",
            str(args.threshold),
            "--top-n",
            str(args.top_n),
            "--title",
            "O15156 / ZBTB7B Predicted Kla Sites (Lab-Weighted E4 + 5D MetaModel)",
            "--x-label",
            "ZBTB7B Residue Position",
        ]
    )

    print(f"Saved predictions to {output_path}")
    print(f"Saved lollipop plot to {figure_path}")


if __name__ == "__main__":
    main()
