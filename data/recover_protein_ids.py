"""Recover UniProt accessions for the anonymized HybridKla training peptides.

Positive samples are matched against HybridKla Supplementary Table S1. Negative
samples are matched against every lysine-centred 51-aa window generated from
the full UniProt sequences of the proteins listed in Table S1.

Ambiguous matches are retained rather than resolved arbitrarily. Proteins
connected by an ambiguous peptide are assigned the same Group_ID, which keeps
all possible source proteins in one cross-validation fold.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd
import requests
from sklearn.model_selection import StratifiedGroupKFold


USER_AGENT = "HybridKla-ID-recovery/1.0"


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union_all(self, items: Iterable[str]) -> None:
        values = sorted(set(items))
        if not values:
            return
        root = self.find(values[0])
        for item in values[1:]:
            other = self.find(item)
            if root != other:
                keep, merge = sorted((root, other))
                self.parent[merge] = keep
                root = keep


def normalize_sequence(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def extract_window(sequence: str, position_1based: int, window_size: int) -> str:
    half = window_size // 2
    center = position_1based - 1
    start = center - half
    end = center + half + 1
    left_padding = "*" * max(0, -start)
    right_padding = "*" * max(0, end - len(sequence))
    return left_padding + sequence[max(0, start) : min(len(sequence), end)] + right_padding


def parse_fasta_records(text: str) -> Dict[str, str]:
    records: Dict[str, List[str]] = {}
    accession = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].split()[0]
            parts = header.split("|")
            accession = parts[1] if len(parts) >= 3 else parts[0]
            records.setdefault(accession, [])
        elif accession:
            records[accession].append(line)
    return {key: normalize_sequence("".join(value)) for key, value in records.items()}


def read_cached_fastas(cache_dir: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    if not cache_dir.exists():
        return sequences
    for fasta_path in cache_dir.glob("*.fasta"):
        parsed = parse_fasta_records(fasta_path.read_text(encoding="utf-8"))
        if fasta_path.stem in parsed:
            sequences[fasta_path.stem] = parsed[fasta_path.stem]
        elif len(parsed) == 1:
            sequences[fasta_path.stem] = next(iter(parsed.values()))
    return sequences


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def download_uniprot_sequences(
    accessions: Sequence[str],
    cache_dir: Path,
    seed_cache_dir: Path | None,
    batch_size: int,
    retries: int,
) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = read_cached_fastas(cache_dir)
    if seed_cache_dir is not None:
        for accession, sequence in read_cached_fastas(seed_cache_dir).items():
            cached.setdefault(accession, sequence)
    missing = [accession for accession in accessions if accession not in cached]
    print(f"UniProt sequences: {len(cached)} cached, {len(missing)} to download")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    endpoint = "https://rest.uniprot.org/uniprotkb/accessions"

    for batch_number, batch in enumerate(chunks(missing, batch_size), start=1):
        response = None
        for attempt in range(retries):
            try:
                response = session.get(
                    endpoint,
                    params={"accessions": ",".join(batch), "format": "fasta"},
                    timeout=90,
                )
            except requests.RequestException as exc:
                print(
                    f"Warning: batch {batch_number}, attempt {attempt + 1} "
                    f"network error: {exc}"
                )
                time.sleep(min(2**attempt, 10))
                continue
            if response.status_code == 200:
                break
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", "3"))
            else:
                wait = 2**attempt
            time.sleep(wait)
        if response is None or response.status_code != 200:
            status = "no response" if response is None else response.status_code
            print(f"Warning: UniProt batch {batch_number} failed ({status})")
            continue

        parsed = parse_fasta_records(response.text)
        cached.update(parsed)
        for accession, sequence in parsed.items():
            fasta_path = cache_dir / f"{accession}.fasta"
            fasta_path.write_text(f">{accession}\n{sequence}\n", encoding="utf-8")
        print(
            f"Downloaded batch {batch_number}: "
            f"{len(parsed)}/{len(batch)} sequences (total available {len(cached)})"
        )

    return cached


def load_supplement(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=1)
    if raw.shape[1] < 4:
        raise ValueError("Supplementary Table S1 must contain four columns")
    supplement = raw.iloc[:, :4].copy()
    supplement.columns = ["UniProt_ID", "Site", "Sequence", "Species"]
    supplement = supplement.dropna(subset=["UniProt_ID", "Sequence"])
    supplement["UniProt_ID"] = supplement["UniProt_ID"].astype(str).str.strip()
    supplement["Site"] = pd.to_numeric(supplement["Site"], errors="coerce").astype("Int64")
    supplement["Sequence"] = supplement["Sequence"].map(normalize_sequence)
    supplement["Species"] = supplement["Species"].astype(str).str.strip()
    return supplement


def make_match_index(
    supplement: pd.DataFrame,
    protein_sequences: Dict[str, str],
    window_size: int,
) -> Tuple[
    Dict[str, Set[Tuple[str, int]]],
    Dict[str, Set[Tuple[str, int]]],
    Dict[str, str],
]:
    positive_index: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
    species_by_accession: Dict[str, str] = {}
    for row in supplement.itertuples(index=False):
        if pd.isna(row.Site):
            continue
        positive_index[row.Sequence].add((row.UniProt_ID, int(row.Site)))
        species_by_accession.setdefault(row.UniProt_ID, row.Species)

    all_k_index: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
    for accession, sequence in protein_sequences.items():
        for index, residue in enumerate(sequence):
            if residue == "K":
                position = index + 1
                peptide = extract_window(sequence, position, window_size)
                all_k_index[peptide].add((accession, position))

    return positive_index, all_k_index, species_by_accession


def candidate_text(matches: Set[Tuple[str, int]]) -> Tuple[str, str]:
    ordered = sorted(matches)
    accessions = ";".join(sorted({accession for accession, _ in ordered}))
    sites = ";".join(f"{accession}:K{site}" for accession, site in ordered)
    return accessions, sites


def recover_rows(
    train_df: pd.DataFrame,
    positive_index: Dict[str, Set[Tuple[str, int]]],
    all_k_index: Dict[str, Set[Tuple[str, int]]],
    species_by_accession: Dict[str, str],
    folds: int,
    seed: int,
) -> pd.DataFrame:
    union_find = UnionFind()
    candidate_sets: List[Set[Tuple[str, int]]] = []

    for row in train_df.itertuples(index=False):
        sequence = normalize_sequence(row.Sequence)
        matches = positive_index.get(sequence, set()) if int(row.Label) == 1 else all_k_index.get(sequence, set())
        matches = set(matches)
        candidate_sets.append(matches)
        union_find.union_all(accession for accession, _ in matches)

    recovered_rows: List[Dict[str, object]] = []
    for row_number, (row, matches) in enumerate(
        zip(train_df.itertuples(index=False), candidate_sets),
        start=1,
    ):
        sample_id = str(getattr(row, "UniProt_ID", f"train{row_number}"))
        accessions = sorted({accession for accession, _ in matches})
        accession_text, site_text = candidate_text(matches)
        unique_accession = accessions[0] if len(accessions) == 1 else pd.NA
        positions = sorted({site for _, site in matches})
        unique_site = positions[0] if len(matches) == 1 else pd.NA
        if len(accessions) == 1:
            status = "exact_unique"
            group_id = union_find.find(accessions[0])
        elif len(accessions) > 1:
            status = "exact_ambiguous"
            group_id = union_find.find(accessions[0])
        else:
            status = "unresolved"
            group_id = f"UNRESOLVED_{sample_id}"

        species = sorted(
            {
                species_by_accession[accession]
                for accession in accessions
                if accession in species_by_accession
            }
        )
        recovered_rows.append(
            {
                "Sample_ID": sample_id,
                "UniProt_ID": unique_accession,
                "Site": unique_site,
                "Sequence": normalize_sequence(row.Sequence),
                "Label": int(row.Label),
                "Species": ";".join(species) if species else pd.NA,
                "Match_Status": status,
                "Candidate_IDs": accession_text if accession_text else pd.NA,
                "Candidate_Sites": site_text if site_text else pd.NA,
                "Group_ID": group_id,
            }
        )

    recovered = pd.DataFrame(recovered_rows)
    recovered["Fold"] = 0
    resolved = recovered["Match_Status"] != "unresolved"
    resolved_index = recovered.index[resolved].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(
        splitter.split(
            recovered.loc[resolved, "Sequence"],
            recovered.loc[resolved, "Label"],
            recovered.loc[resolved, "Group_ID"],
        ),
        start=1,
    ):
        recovered.loc[resolved_index[test_index], "Fold"] = fold

    group_fold_counts = recovered.loc[resolved].groupby("Group_ID")["Fold"].nunique()
    if (group_fold_counts > 1).any():
        raise RuntimeError("Group leakage detected while generating folds")
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train/train_data.xlsx", type=Path)
    parser.add_argument(
        "--supplement",
        default="data/HybridKla_table_s1.xlsx",
        type=Path,
    )
    parser.add_argument("--output", default="train/train_data_with_ids.xlsx", type=Path)
    parser.add_argument("--cache-dir", default="data/uniprot_recovery_cache", type=Path)
    parser.add_argument("--seed-cache-dir", default="data/uniprot_cache", type=Path)
    parser.add_argument("--window-size", default=51, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--retries", default=4, type=int)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()

    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd")

    train_df = pd.read_excel(args.train)
    required = {"Sequence", "Label"}
    missing_columns = required - set(train_df.columns)
    if missing_columns:
        raise ValueError(f"Training table is missing columns: {sorted(missing_columns)}")

    supplement = load_supplement(args.supplement)
    accessions = sorted(supplement["UniProt_ID"].unique())
    protein_sequences = read_cached_fastas(args.cache_dir)
    for accession, sequence in read_cached_fastas(args.seed_cache_dir).items():
        protein_sequences.setdefault(accession, sequence)
    if args.download_missing:
        protein_sequences = download_uniprot_sequences(
            accessions,
            args.cache_dir,
            args.seed_cache_dir,
            args.batch_size,
            args.retries,
        )
    protein_sequences = {
        accession: protein_sequences[accession]
        for accession in accessions
        if accession in protein_sequences
    }

    positive_index, all_k_index, species_by_accession = make_match_index(
        supplement,
        protein_sequences,
        args.window_size,
    )
    recovered = recover_rows(
        train_df,
        positive_index,
        all_k_index,
        species_by_accession,
        args.folds,
        args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_excel(args.output, index=False)

    print(f"Wrote {len(recovered)} samples to {args.output}")
    print(recovered["Match_Status"].value_counts().to_string())
    print("Fold sizes:")
    print(recovered.groupby("Fold")["Label"].agg(["size", "sum"]).to_string())
    print(f"Unique leakage-safe groups: {recovered['Group_ID'].nunique()}")


if __name__ == "__main__":
    main()
