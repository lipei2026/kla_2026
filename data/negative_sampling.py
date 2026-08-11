"""
同蛋白内负样本采样脚本（乳酸化/K位点）。

输入：一个表格（xlsx/csv/tsv），包含至少两类信息：
1) 蛋白标识列（protein id）
2) 蛋白序列列（full sequence）
3) 乳酸化阳性位点列（position）。支持：
   - 一蛋白一行：pos列为“多个位点”的字符串（如 12;45;130）
   - 一位点一行：同一蛋白多行，每行一个pos

输出：用于本仓库训练的窗口样本表（默认 xlsx）：
- UniProt_ID：蛋白标识
- Sequence：51aa窗口（中心为该位点K，越界用*补齐）
- Label：1(阳性) / 0(阴性)

示例：
python scripts/intra_protein_negative_sampling.py ^
  --input your.xlsx --output train_data.xlsx ^
  --protein-id-col UniProt_ID --seq-col Sequence --pos-col Kla_sites ^
  --pos-base 1 --neg-per-pos 2 --min-distance 10 --window 51 --seed 42
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _read_table(path: str):
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported input file extension: {ext}")


def _write_table(df, path: str):
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    if ext == ".csv":
        df.to_csv(path, index=False)
        return
    if ext in {".tsv", ".txt"}:
        df.to_csv(path, index=False, sep="\t")
        return
    raise ValueError(f"Unsupported output file extension: {ext}")


def _normalize_seq(seq: str) -> str:
    s = (seq or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def _parse_positions(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
        try:
            return [int(value)]
        except Exception:
            return []
    s = str(value).strip()
    if not s:
        return []
    m_k = re.findall(r"_K(\d+)\b", s)
    if m_k:
        return [int(x) for x in m_k]
    s = s.replace("，", ",").replace("；", ";")
    parts = re.split(r"[;,|/\s]+", s)
    out: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"-?\d+", p)
        if not m:
            continue
        out.append(int(m.group(0)))
    return out


def _extract_window(seq: str, pos_1based: int, window: int, pad: str = "*") -> str:
    if window % 2 == 0:
        raise ValueError("window must be odd")
    half = window // 2
    left = pos_1based - half
    right = pos_1based + half
    out = []
    for i in range(left, right + 1):
        if i < 1 or i > len(seq):
            out.append(pad)
        else:
            out.append(seq[i - 1])
    return "".join(out)


@dataclass
class ProteinRecord:
    protein_id: str
    sequence: str
    pos_sites_1based: Set[int]


def _parse_fasta(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    seq_lines = [ln for ln in lines if not ln.startswith(">")]
    return _normalize_seq("".join(seq_lines))


def _fetch_uniprot_fasta(
    accession: str,
    timeout_s: int,
    retries: int,
) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    last_err: Optional[Exception] = None
    for attempt in range(max(retries, 1)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hybridkla-sampler/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            seq = _parse_fasta(text)
            return seq if seq else None
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                wait_s = int(retry_after) if retry_after and retry_after.isdigit() else 2
                time.sleep(wait_s)
                continue
            if e.code == 404:
                return None
            time.sleep(1)
        except Exception as e:
            last_err = e
            time.sleep(1)
    if last_err is not None:
        print(f"Failed to fetch UniProt sequence for {accession}: {last_err}", file=sys.stderr)
    return None


def _load_uniprot_cache(cache_dir: str) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not cache_dir:
        return cache
    if not os.path.isdir(cache_dir):
        return cache
    for fn in os.listdir(cache_dir):
        if not fn.endswith(".fasta"):
            continue
        acc = fn[:-6]
        try:
            with open(os.path.join(cache_dir, fn), "r", encoding="utf-8") as f:
                seq = _parse_fasta(f.read())
            if seq:
                cache[acc] = seq
        except Exception:
            continue
    return cache


def _save_uniprot_cache(cache_dir: str, accession: str, sequence: str) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{accession}.fasta")
    if os.path.exists(path):
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f">{accession}\n")
            for i in range(0, len(sequence), 60):
                f.write(sequence[i : i + 60] + "\n")
    except Exception:
        return


def _collect_proteins(
    df,
    protein_id_col: str,
    seq_col: str,
    pos_col: str,
    pos_base: int,
    fetch_uniprot: bool,
    uniprot_timeout_s: int,
    uniprot_retries: int,
    uniprot_cache_dir: str,
) -> Dict[str, ProteinRecord]:
    proteins: Dict[str, ProteinRecord] = {}
    seq_cache = _load_uniprot_cache(uniprot_cache_dir)

    for _, row in df.iterrows():
        pid = str(row.get(protein_id_col, "")).strip()
        if not pid:
            continue
        seq = ""
        if seq_col and seq_col in df.columns:
            seq = _normalize_seq(row.get(seq_col, ""))
        if not seq and fetch_uniprot:
            if pid in seq_cache:
                seq = seq_cache[pid]
            else:
                seq = _fetch_uniprot_fasta(pid, timeout_s=uniprot_timeout_s, retries=uniprot_retries) or ""
                if seq:
                    seq_cache[pid] = seq
                    _save_uniprot_cache(uniprot_cache_dir, pid, seq)
        if not seq:
            print(f"Skip {pid}: missing sequence", file=sys.stderr)
            continue

        positions_raw = _parse_positions(row.get(pos_col, None))
        positions_1based: Set[int] = set()
        for p in positions_raw:
            if pos_base == 0:
                p = p + 1
            if p <= 0:
                continue
            positions_1based.add(p)

        if pid not in proteins:
            proteins[pid] = ProteinRecord(protein_id=pid, sequence=seq, pos_sites_1based=set())
        else:
            if proteins[pid].sequence != seq:
                raise ValueError(f"Conflicting sequences for protein_id={pid}")

        proteins[pid].pos_sites_1based.update(positions_1based)

    return proteins


def _k_positions(seq: str) -> List[int]:
    return [i + 1 for i, aa in enumerate(seq) if aa == "K"]


def _sample_negatives_within_protein(
    rng,
    k_positions_1based: Sequence[int],
    pos_sites_1based: Set[int],
    neg_count: int,
    min_distance: int,
) -> List[int]:
    if neg_count <= 0:
        return []

    pos_list = sorted(pos_sites_1based)
    candidates_strict: List[int] = []
    for kp in k_positions_1based:
        if kp in pos_sites_1based:
            continue
        ok = True
        if min_distance > 0 and pos_list:
            for pp in pos_list:
                if abs(kp - pp) <= min_distance:
                    ok = False
                    break
        if ok:
            candidates_strict.append(kp)

    candidates_relaxed = [kp for kp in k_positions_1based if kp not in pos_sites_1based]

    if len(candidates_strict) >= neg_count:
        return rng.sample(candidates_strict, neg_count)

    if len(candidates_relaxed) >= neg_count:
        return rng.sample(candidates_relaxed, neg_count)

    if not candidates_relaxed:
        return []

    return [rng.choice(candidates_relaxed) for _ in range(neg_count)]


def build_dataset(
    proteins: Dict[str, ProteinRecord],
    window: int,
    neg_per_pos: int,
    min_distance: int,
    seed: int,
    require_center_k: bool,
):
    import pandas as pd
    import random

    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    for pid, rec in proteins.items():
        seq = rec.sequence
        kpos = _k_positions(seq)
        if not kpos:
            continue

        pos_sites = {p for p in rec.pos_sites_1based if 1 <= p <= len(seq)}
        if require_center_k:
            pos_sites = {p for p in pos_sites if seq[p - 1] == "K"}

        for p in sorted(pos_sites):
            win = _extract_window(seq, p, window=window)
            rows.append({"UniProt_ID": pid, "Site": p, "Sequence": win, "Label": 1})

        if neg_per_pos > 0 and pos_sites:
            neg_total = len(pos_sites) * neg_per_pos
            neg_sites = _sample_negatives_within_protein(
                rng=rng,
                k_positions_1based=kpos,
                pos_sites_1based=pos_sites,
                neg_count=neg_total,
                min_distance=min_distance,
            )
            for p in neg_sites:
                win = _extract_window(seq, p, window=window)
                rows.append({"UniProt_ID": pid, "Site": p, "Sequence": win, "Label": 0})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protein-id-col", default="UniProt_ID")
    parser.add_argument("--seq-col", default="")
    parser.add_argument("--pos-col", required=True)
    parser.add_argument("--pos-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--window", type=int, default=51)
    parser.add_argument("--neg-per-pos", type=int, default=2)
    parser.add_argument("--min-distance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-center-k", action="store_true")
    parser.add_argument("--fetch-uniprot", action="store_true")
    parser.add_argument("--uniprot-timeout", type=int, default=30)
    parser.add_argument("--uniprot-retries", type=int, default=3)
    parser.add_argument("--uniprot-cache-dir", default="")
    args = parser.parse_args()

    df = _read_table(args.input)
    proteins = _collect_proteins(
        df=df,
        protein_id_col=args.protein_id_col,
        seq_col=args.seq_col,
        pos_col=args.pos_col,
        pos_base=args.pos_base,
        fetch_uniprot=args.fetch_uniprot,
        uniprot_timeout_s=args.uniprot_timeout,
        uniprot_retries=args.uniprot_retries,
        uniprot_cache_dir=args.uniprot_cache_dir,
    )

    out_df = build_dataset(
        proteins=proteins,
        window=args.window,
        neg_per_pos=args.neg_per_pos,
        min_distance=args.min_distance,
        seed=args.seed,
        require_center_k=args.require_center_k,
    )

    if out_df.empty:
        raise RuntimeError("No samples generated. Check input columns and positions.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    _write_table(out_df, args.output)


if __name__ == "__main__":
    main()