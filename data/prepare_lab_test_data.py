from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw_data_lab.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "lab_test_data.xlsx"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "uniprot_cache"
WINDOW_SIZE = 51


def normalize_sequence(seq: str) -> str:
    return re.sub(r"\s+", "", (seq or "").strip().upper())


def parse_site_from_index(index_value: str) -> int | None:
    match = re.search(r"_K(\d+)\b", str(index_value))
    return int(match.group(1)) if match else None


def fetch_uniprot_sequence(accession: str, timeout_s: int = 20, retries: int = 3) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    last_error: Exception | None = None
    for _ in range(max(retries, 1)):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "hybridkla-lab-test/1.0"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                fasta_text = response.read().decode("utf-8", errors="replace")
            lines = [line.strip() for line in fasta_text.splitlines() if line.strip() and not line.startswith(">")]
            sequence = normalize_sequence("".join(lines))
            return sequence or None
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                return None
            time.sleep(1)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            time.sleep(1)
    if last_error is not None:
        print(f"Failed to fetch {accession}: {last_error}", file=sys.stderr)
    return None


def load_cached_sequence(accession: str, cache_dir: Path) -> str | None:
    fasta_path = cache_dir / f"{accession}.fasta"
    if not fasta_path.exists():
        return None
    lines = [line.strip() for line in fasta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sequence = "".join(line for line in lines if not line.startswith(">"))
    return normalize_sequence(sequence) or None


def save_cached_sequence(accession: str, sequence: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = cache_dir / f"{accession}.fasta"
    if fasta_path.exists():
        return
    with fasta_path.open("w", encoding="utf-8") as handle:
        handle.write(f">{accession}\n")
        for start in range(0, len(sequence), 60):
            handle.write(sequence[start : start + 60] + "\n")


def extract_window(sequence: str, site_1based: int, window_size: int = WINDOW_SIZE, pad: str = "*") -> str:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    half_window = window_size // 2
    left = site_1based - half_window
    right = site_1based + half_window
    output = []
    for position in range(left, right + 1):
        if position < 1 or position > len(sequence):
            output.append(pad)
        else:
            output.append(sequence[position - 1])
    return "".join(output)


def build_window_from_peptide(peptide: str, window_size: int = WINDOW_SIZE, pad: str = "*") -> str | None:
    peptide = (peptide or "").strip()
    if not peptide:
        return None

    center_index = None
    for index, amino_acid in enumerate(peptide):
        if amino_acid == "k":
            center_index = index
            break

    peptide_upper = peptide.upper()
    if center_index is None:
        lys_positions = [idx for idx, aa in enumerate(peptide_upper) if aa == "K"]
        if len(lys_positions) == 1:
            center_index = lys_positions[0]
        else:
            return None

    half_window = window_size // 2
    left_pad = max(0, half_window - center_index)
    right_pad = max(0, window_size - left_pad - len(peptide_upper))
    padded = (pad * left_pad) + peptide_upper + (pad * right_pad)
    start = center_index + left_pad - half_window
    return padded[start : start + window_size]


def resolve_sequence(accession: str, cache_dir: Path, timeout_s: int, retries: int) -> str | None:
    cached = load_cached_sequence(accession, cache_dir)
    if cached:
        return cached
    fetched = fetch_uniprot_sequence(accession, timeout_s=timeout_s, retries=retries)
    if fetched:
        save_cached_sequence(accession, fetched, cache_dir)
    return fetched


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw_data_lab.xlsx to standardized 51 aa lab test windows.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to raw laboratory Excel file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to standardized output xlsx/csv.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory for cached UniProt FASTA files.")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="Window size, must be odd.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick testing.")
    parser.add_argument(
        "--allow-peptide-fallback",
        action="store_true",
        help="If UniProt sequence fetch fails, use modified peptide to build a padded pseudo-window.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    df = pd.read_excel(input_path)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    records = []
    skipped = []

    for _, row in df.iterrows():
        record_id = str(row.get("Index", "")).strip()
        accession = str(row.get("Accession", "")).strip()
        peptide = str(row.get("Peptide", "")).strip()
        site = parse_site_from_index(record_id)

        if not accession or site is None:
            skipped.append((record_id, "missing accession or site"))
            continue

        full_sequence = resolve_sequence(accession, cache_dir=cache_dir, timeout_s=20, retries=3)
        sequence_source = "uniprot"

        if full_sequence:
            if site < 1 or site > len(full_sequence):
                skipped.append((record_id, f"site {site} out of range for sequence length {len(full_sequence)}"))
                continue
            if full_sequence[site - 1] != "K":
                skipped.append((record_id, f"site {site} is {full_sequence[site - 1]} instead of K"))
                continue
            window = extract_window(full_sequence, site, window_size=args.window)
        elif args.allow_peptide_fallback:
            window = build_window_from_peptide(peptide, window_size=args.window)
            sequence_source = "peptide_fallback"
            if window is None:
                skipped.append((record_id, "failed to build peptide fallback window"))
                continue
        else:
            skipped.append((record_id, "failed to fetch UniProt sequence"))
            continue

        records.append(
            {
                "pepname": record_id,
                "UniProt_ID": accession,
                "Site": site,
                "Sequence": window,
                "Label": 1,
                "Sequence_Source": sequence_source,
                "Gene": row.get("Gene", ""),
                "Description": row.get("Description", ""),
                "Raw_Peptide": peptide,
            }
        )

    output_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        output_df.to_excel(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        output_df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must end with .xlsx, .xls or .csv")

    print(f"Saved {len(output_df)} standardized lab test records to {output_path}")
    print(f"Skipped {len(skipped)} rows")
    if skipped:
        preview = skipped[:10]
        print("First skipped rows:")
        for record_id, reason in preview:
            print(f"- {record_id}: {reason}")


if __name__ == "__main__":
    main()
