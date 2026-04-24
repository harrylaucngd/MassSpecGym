"""
Evaluate precomputed MIST fingerprints for retrieval on MSG test set.

Supports four evaluation modes:
  top1          - rank-1 MIST-CF formula fingerprint only
  top5_avg      - element-wise mean of rank 1..5 fps, then Tanimoto vs candidates
  top5_max      - element-wise max (OR) of rank 1..5 fps, then Tanimoto vs candidates
  top5_rank_avg - per-rank Tanimoto scores → candidate ranks → average rank across 5

Usage:
    python scripts/eval_precomputed_fps.py \
        --scratch-dir /home/mrunali/orcd/scratch/msg_benchmark \
        --labels data/msg/labels.tsv \
        --split data/msg/split.tsv \
        --candidates ~/ms-data/data/msg/MassSpecGym_retrieval_candidates_mass.json \
        [--modes top1 top5_avg top5_max top5_rank_avg] \
        [--max-ranks 5] [--hits 1 5 10 20] [--also-gt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mol_to_fp(smiles: str, fp_size: int = 4096, radius: int = 2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    return np.frombuffer(bv.ToBitString().encode(), dtype="u1") - ord("0")


def smiles_to_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    if inchi is None:
        return None
    return Chem.InchiToInchiKey(inchi)


def tanimoto_batch(query: np.ndarray, cand_matrix: np.ndarray) -> np.ndarray:
    """Generalized Tanimoto — works for binary or averaged (float) query vectors.

    query: (D,) float, cand_matrix: (N, D) float → scores: (N,)
    """
    inter = cand_matrix @ query                         # (N,)
    union = query.sum() + cand_matrix.sum(axis=1) - inter
    return inter / np.maximum(union, 1e-8)


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert descending scores to 1-based ranks (rank 1 = highest score)."""
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Per-spectrum aggregation
# ---------------------------------------------------------------------------

def aggregate_fps(fps: list[np.ndarray], mode: str) -> np.ndarray:
    """Combine multiple fingerprints (one per rank) into a single query vector."""
    stack = np.stack(fps, axis=0).astype(np.float32)   # (K, D)
    if mode == "top5_avg":
        return stack.mean(axis=0)                       # (D,) float in [0, 1]
    elif mode == "top5_max":
        return stack.max(axis=0)                        # (D,) binary {0, 1}
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    mode: str,
    rank_fps: dict[int, dict[str, np.ndarray]],  # rank → spec_id → fp
    df: pd.DataFrame,
    spec2smiles: dict[str, str],
    candidates_json: dict[str, list[str]],
    fp_cache: dict[str, np.ndarray | None],
    ik_cache: dict[str, str | None],
    hit_ks: list[int],
    max_ranks: int,
) -> dict:
    available_ranks = sorted(k for k in rank_fps if k <= max_ranks)

    hit_counts = {k: 0 for k in hit_ks}
    tanimoto_top1: list[float] = []
    n_evaluated = 0
    n_skipped_no_fp = 0
    n_skipped_no_cands = 0

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = mol_to_fp(smi)
        return fp_cache[smi]

    def get_ik(smi):
        if smi not in ik_cache:
            ik_cache[smi] = smiles_to_inchikey(smi)
        return ik_cache[smi]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=mode):
        spec_id = str(row["identifier"])
        true_smiles = spec2smiles[spec_id]

        # --- Collect available rank fingerprints for this spectrum -------
        if mode == "top1":
            if 1 not in rank_fps or spec_id not in rank_fps[1]:
                n_skipped_no_fp += 1
                continue
            fps_for_spec = [rank_fps[1][spec_id]]
        else:
            fps_for_spec = [
                rank_fps[k][spec_id]
                for k in available_ranks
                if spec_id in rank_fps[k]
            ]
            if not fps_for_spec:
                n_skipped_no_fp += 1
                continue

        # --- Candidate matrix --------------------------------------------
        if true_smiles not in candidates_json:
            n_skipped_no_cands += 1
            continue
        cand_smiles = candidates_json[true_smiles]

        cand_fps, cand_iks = [], []
        for smi in cand_smiles:
            fp = get_fp(smi)
            ik = get_ik(smi)
            if fp is not None and ik is not None:
                cand_fps.append(fp)
                cand_iks.append(ik)
        if not cand_fps:
            n_skipped_no_cands += 1
            continue

        true_ik = get_ik(true_smiles)
        if true_ik is None:
            n_skipped_no_cands += 1
            continue

        cand_matrix = np.stack(cand_fps, axis=0).astype(np.float32)  # (N, D)

        # --- Score and rank candidates -----------------------------------
        if mode in ("top1", "top5_avg", "top5_max"):
            if mode == "top1":
                query = fps_for_spec[0].astype(np.float32)
            else:
                query = aggregate_fps(fps_for_spec, mode)
            scores = tanimoto_batch(query, cand_matrix)
            order = np.argsort(scores)[::-1]

        elif mode == "top5_rank_avg":
            # For each rank fp: compute Tanimoto scores → convert to ranks.
            # Average ranks across all available fps, then sort ascending.
            rank_matrix = np.stack(
                [scores_to_ranks(tanimoto_batch(fp.astype(np.float32), cand_matrix))
                 for fp in fps_for_spec],
                axis=0,
            ).astype(np.float32)              # (K, N)
            avg_ranks = rank_matrix.mean(axis=0)  # (N,)
            order = np.argsort(avg_ranks)          # ascending = best rank first
            scores = -avg_ranks                    # invert so "best" = highest score for tanimoto@1 reporting

        ranked_iks = [cand_iks[i] for i in order]

        # tanimoto@1: actual Tanimoto of top-1 candidate (always use rank-1 fp for this)
        top1_fp = fps_for_spec[0].astype(np.float32)
        top1_cand_fp = cand_fps[order[0]].astype(np.float32)
        inter = float(np.dot(top1_fp, top1_cand_fp))
        union = float(top1_fp.sum() + top1_cand_fp.sum() - inter)
        tanimoto_top1.append(inter / max(union, 1e-8))

        for hit_k in hit_ks:
            if true_ik in ranked_iks[:hit_k]:
                hit_counts[hit_k] += 1

        n_evaluated += 1

    result = {
        "mode": mode,
        "n_fps_used": 1 if mode == "top1" else len(available_ranks),
        "n_evaluated": n_evaluated,
        "skipped_no_fp": n_skipped_no_fp,
        "skipped_no_cands": n_skipped_no_cands,
    }
    for k in hit_ks:
        result[f"HitRate@{k}"] = hit_counts[k] / max(n_evaluated, 1)
    result["mean_tanimoto@1"] = float(np.mean(tanimoto_top1)) if tanimoto_top1 else 0.0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-dir", type=Path, required=True,
                    help="Dir with fingerprints/ subdir containing fingerprints_mistcf_rank{k}.pt")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True,
                    help="Path to MassSpecGym_retrieval_candidates_mass.json")
    ap.add_argument("--modes", nargs="+",
                    default=["top1", "top5_avg", "top5_max", "top5_rank_avg"],
                    choices=["top1", "top5_avg", "top5_max", "top5_rank_avg"],
                    help="Evaluation modes to run (default: all four)")
    ap.add_argument("--max-ranks", type=int, default=5,
                    help="Maximum number of formula ranks to use for top-5 modes (default: 5)")
    ap.add_argument("--hits", type=int, nargs="+", default=[1, 5, 10, 20],
                    help="K values for HitRate@K")
    ap.add_argument("--also-gt", action="store_true",
                    help="Also run top1 eval on fingerprints_gt.pt as baseline")
    ap.add_argument("--split-fold", type=str, default="test")
    args = ap.parse_args()

    fp_dir = args.scratch_dir / "fingerprints"

    # --- Load test set ---------------------------------------------------
    labels = pd.read_csv(args.labels, sep="\t")
    split = pd.read_csv(args.split, sep="\t")

    spec_col = "spec" if "spec" in labels.columns else "identifier"
    labels = labels.rename(columns={spec_col: "identifier"})
    split_spec_col = "spec" if "spec" in split.columns else "name"
    fold_col = "fold" if "fold" in split.columns else "split"
    split = split.rename(columns={split_spec_col: "identifier", fold_col: "fold"})

    df = labels.merge(split[["identifier", "fold"]], on="identifier", how="inner")
    df = df[df["fold"] == args.split_fold].reset_index(drop=True)
    print(f"Test set ({args.split_fold}): {len(df)} spectra")

    spec2smiles = dict(zip(df["identifier"].astype(str), df["smiles"].astype(str)))

    # --- Load candidates JSON --------------------------------------------
    print(f"Loading candidates from {args.candidates} ...")
    with open(args.candidates) as f:
        candidates_json: dict[str, list[str]] = json.load(f)
    print(f"  {len(candidates_json)} query SMILES in candidate set")

    # --- Load all rank fingerprints upfront ------------------------------
    rank_fps: dict[int, dict[str, np.ndarray]] = {}
    for k in range(1, args.max_ranks + 1):
        pth = fp_dir / f"fingerprints_mistcf_rank{k}.pt"
        if pth.exists():
            raw = torch.load(pth, map_location="cpu", weights_only=False)
            rank_fps[k] = {sid: t.numpy() for sid, t in raw.items()}
            print(f"  Loaded rank {k}: {len(rank_fps[k])} spectra")
        else:
            print(f"  [WARN] Missing: {pth.name}")

    gt_fps: dict[str, np.ndarray] | None = None
    if args.also_gt:
        for name in ("fingerprints_gt.pt", "fingerprints_mistcf_gt.pt"):
            pth = fp_dir / name
            if pth.exists():
                raw = torch.load(pth, map_location="cpu", weights_only=False)
                gt_fps = {sid: t.numpy() for sid, t in raw.items()}
                print(f"  Loaded GT fingerprints: {len(gt_fps)} spectra")
                break
        if gt_fps is None:
            print("  [WARN] --also-gt: no GT fingerprint file found in", fp_dir)

    if not rank_fps:
        print("No fingerprint files found. Exiting.")
        return

    # Shared caches (shared across all modes to avoid redundant RDKit calls)
    fp_cache: dict[str, np.ndarray | None] = {}
    ik_cache: dict[str, str | None] = {}

    # --- Run evaluations -------------------------------------------------
    modes_to_run = list(args.modes)
    if args.also_gt and gt_fps is not None:
        modes_to_run = ["gt_top1"] + modes_to_run

    all_results: list[dict] = []

    for mode in modes_to_run:
        print(f"\n{'='*60}\n{mode}\n{'='*60}")

        if mode == "gt_top1":
            rank_fps_input = {1: gt_fps}
            result = evaluate("top1", rank_fps_input, df, spec2smiles,
                              candidates_json, fp_cache, ik_cache, args.hits, 1)
            result["mode"] = "gt_top1"
        else:
            result = evaluate(mode, rank_fps, df, spec2smiles,
                              candidates_json, fp_cache, ik_cache, args.hits, args.max_ranks)

        all_results.append(result)

        print(f"  Evaluated: {result['n_evaluated']}  "
              f"skipped (no fp): {result['skipped_no_fp']}  "
              f"skipped (no cands): {result['skipped_no_cands']}")
        for k in args.hits:
            print(f"  HitRate@{k}: {result[f'HitRate@{k}']:.4f}")
        print(f"  Mean Tanimoto@1 (rank-1 fp vs top cand): {result['mean_tanimoto@1']:.4f}")

    # --- Summary table ---------------------------------------------------
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    results_df = pd.DataFrame(all_results)
    hit_cols = [f"HitRate@{k}" for k in args.hits] + ["mean_tanimoto@1"]
    print(results_df[["mode", "n_fps_used", "n_evaluated"] + hit_cols].to_string(
        index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
