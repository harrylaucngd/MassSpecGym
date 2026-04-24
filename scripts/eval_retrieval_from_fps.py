#!/usr/bin/env python3
"""
Evaluate retrieval using pre-computed MIST fingerprints.

Loads fingerprints from .pt files, computes Tanimoto or cosine similarity
against MassSpecGym retrieval candidates, and reports HitRate@K. Optionally
binarizes predicted fingerprints at --threshold before scoring.

Usage:
    python scripts/eval_retrieval_from_fps.py \
        --fp-dir /path/to/fingerprints \
        --labels /path/to/labels.tsv \
        --split /path/to/split.tsv \
        --candidates-json /path/to/MassSpecGym_retrieval_candidates_mass.json \
        --output-dir /path/to/results \
        [--similarity tanimoto|cosine] [--threshold 0.187]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm


def _load_fp_cache(path: Path, n_bits: int) -> tuple[dict, dict]:
    """Return (fp_cache: smi->float32 fp, ik_cache: smi->inchikey|None)."""
    if not path.exists():
        return {}, {}
    data = np.load(path, allow_pickle=True)
    smis = data["smiles"].tolist()
    fps_packed = data["fps"]  # (N, ceil(n_bits/8)) uint8, bit-packed
    fps = np.unpackbits(fps_packed, axis=1)[:, :n_bits].astype(np.float32)
    fp_cache = {s: fps[i] for i, s in enumerate(smis)}
    ik_cache = {}
    if "inchikeys" in data.files:
        iks = data["inchikeys"].tolist()
        ik_cache = {s: (ik if ik else None) for s, ik in zip(smis, iks)}
    print(f"  loaded {len(fp_cache)} cached fps + {len(ik_cache)} inchikeys from {path}")
    return fp_cache, ik_cache


def _save_fp_cache(path: Path, fp_cache: dict, ik_cache: dict, n_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    smis = list(fp_cache.keys())
    fps = np.stack([fp_cache[s].astype(np.uint8) for s in smis])
    fps_packed = np.packbits(fps, axis=1)
    iks = np.array([ik_cache.get(s) or "" for s in smis])
    np.savez_compressed(path, smiles=np.array(smis), fps=fps_packed, inchikeys=iks)
    print(f"  saved {len(smis)} fps + inchikeys to {path}")


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 4096) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_similarity(fp_pred: np.ndarray, fp_cands: np.ndarray) -> np.ndarray:
    """Vectorized Tanimoto: fp_pred (D,) vs fp_cands (N, D) -> scores (N,)"""
    intersection = (fp_pred * fp_cands).sum(axis=1)
    union = fp_pred.sum() + fp_cands.sum(axis=1) - intersection
    return intersection / np.maximum(union, 1e-8)


def cosine_similarity_np(fp_pred: np.ndarray, fp_cands: np.ndarray) -> np.ndarray:
    """Vectorized cosine: fp_pred (D,) vs fp_cands (N, D) -> scores (N,)"""
    num = fp_cands @ fp_pred
    denom = np.linalg.norm(fp_cands, axis=1) * np.linalg.norm(fp_pred) + 1e-8
    return num / denom


def compute_hit_rate(scores: np.ndarray, labels: np.ndarray, top_k: int) -> float:
    top_k_idx = np.argsort(scores)[::-1][:top_k]
    return float(labels[top_k_idx].any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-dir", type=Path, required=True,
                        help="Dir with fingerprints_*.pt files")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidates-json", type=Path, required=True,
                        help="MassSpecGym retrieval candidates JSON (keyed by SMILES)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--at-ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--similarity", choices=["tanimoto", "cosine"], default="tanimoto",
                        help="Similarity metric used to rank candidates.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="If set, binarize predicted fps at this threshold before scoring "
                             "(continuous if not set).")
    parser.add_argument("--fp-cache", type=Path, default=None,
                        help="Path to .npz cache for candidate Morgan fps + inchikeys. "
                             "Loaded if present, updated and re-saved if any new SMILES were computed.")
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    labels = pd.read_csv(args.labels, sep="\t")
    col = "spec" if "spec" in labels.columns else "identifier"
    labels = labels.rename(columns={col: "identifier"})

    split = pd.read_csv(args.split, sep="\t")
    if "name" in split.columns:
        split = split.rename(columns={"name": "identifier", "split": "fold"})
    elif "spec" in split.columns:
        split = split.rename(columns={"spec": "identifier"})

    df = labels.merge(split[["identifier", "fold"]], on="identifier", how="inner")
    df = df[df["fold"] == "test"]
    spec_to_smiles = dict(zip(df["identifier"].astype(str), df["smiles"].astype(str)))
    print(f"Test set: {len(df)} spectra")

    # Load candidates
    print(f"Loading candidates from {args.candidates_json}...")
    t0 = time.time()
    with open(args.candidates_json) as f:
        candidates_raw = json.load(f)
    print(f"  Loaded {len(candidates_raw)} candidate sets in {time.time()-t0:.1f}s")

    # Pre-compute candidate fingerprints (cache by SMILES) + inchikeys.
    # Include true smiles too so inchikey/fp are cached for labels.
    print("Collecting unique candidate + query SMILES...")
    all_cand_smiles = set(spec_to_smiles.values())
    for cands in candidates_raw.values():
        all_cand_smiles.update(cands)
    print(f"  {len(all_cand_smiles)} unique SMILES")

    fp_cache: dict = {}
    ik_cache: dict = {}
    if args.fp_cache is not None:
        fp_cache, ik_cache = _load_fp_cache(args.fp_cache, args.n_bits)

    missing = [s for s in all_cand_smiles if s not in fp_cache]
    print(f"  computing {len(missing)} new Morgan fps (n_bits={args.n_bits}, radius={args.radius})")
    for smi in tqdm(missing, desc="Morgan fps"):
        fp_cache[smi] = smiles_to_morgan_fp(smi, radius=args.radius, n_bits=args.n_bits)

    missing_ik = [s for s in all_cand_smiles if s not in ik_cache]
    if missing_ik:
        print(f"  computing {len(missing_ik)} new InChIKeys")
        for smi in tqdm(missing_ik, desc="InChIKeys"):
            mol = Chem.MolFromSmiles(smi)
            ik_cache[smi] = (Chem.inchi.InchiToInchiKey(Chem.MolToInchi(mol))
                             if mol is not None else None)

    if args.fp_cache is not None and (missing or missing_ik):
        _save_fp_cache(args.fp_cache, fp_cache, ik_cache, args.n_bits)

    # Find all fingerprint files
    fp_files = sorted(args.fp_dir.glob("fingerprints_*.pt"))
    if not fp_files:
        print(f"No fingerprint files found in {args.fp_dir}")
        return
    print(f"\nFound {len(fp_files)} fingerprint files:")
    for f in fp_files:
        print(f"  {f.name}")

    # Evaluate each
    all_results = {}
    for fp_path in fp_files:
        rank_label = fp_path.stem.replace("fingerprints_", "")
        print(f"\n{'='*60}")
        print(f"Evaluating: {rank_label}")
        print(f"{'='*60}")

        fps = torch.load(fp_path, map_location="cpu")
        print(f"  Loaded {len(fps)} fingerprints")

        hit_rates = defaultdict(list)
        evaluated = 0
        skipped_no_fp = 0
        skipped_no_cands = 0

        for spec_id, true_smiles in tqdm(spec_to_smiles.items(), desc=rank_label):
            if spec_id not in fps:
                skipped_no_fp += 1
                continue

            if true_smiles not in candidates_raw:
                skipped_no_cands += 1
                continue

            fp_pred = fps[spec_id].numpy().astype(np.float32)
            if args.threshold is not None:
                fp_pred = (fp_pred >= args.threshold).astype(np.float32)
            cand_smiles_list = candidates_raw[true_smiles]

            # Build candidate FP matrix and labels
            cand_fps = np.stack([fp_cache[s] for s in cand_smiles_list])

            # Label each candidate by InChIKey match (from cache).
            true_inchikey = ik_cache.get(true_smiles)
            cand_labels = np.zeros(len(cand_smiles_list), dtype=bool)
            if true_inchikey is not None:
                for i, cs in enumerate(cand_smiles_list):
                    if ik_cache.get(cs) == true_inchikey:
                        cand_labels[i] = True

            if not cand_labels.any():
                skipped_no_cands += 1
                continue

            if args.similarity == "tanimoto":
                scores = tanimoto_similarity(fp_pred, cand_fps)
            else:
                scores = cosine_similarity_np(fp_pred, cand_fps)

            for k in args.at_ks:
                hr = compute_hit_rate(scores, cand_labels, k)
                hit_rates[k].append(hr)

            evaluated += 1

        print(f"  Evaluated: {evaluated}, Skipped (no FP): {skipped_no_fp}, Skipped (no cands): {skipped_no_cands}")

        result = {}
        for k in args.at_ks:
            if hit_rates[k]:
                hr = np.mean(hit_rates[k])
                result[f"HitRate@{k}"] = hr
                print(f"  HitRate@{k}: {hr:.4f}")
            else:
                result[f"HitRate@{k}"] = 0.0
        result["n_evaluated"] = evaluated
        all_results[rank_label] = result

    # Save summary
    tag = f"{args.similarity}" + (f"_thr{args.threshold}" if args.threshold is not None else "_raw")
    summary_path = args.output_dir / f"retrieval_results_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump({"similarity": args.similarity, "threshold": args.threshold,
                   "results": all_results}, f, indent=2)
    print(f"\nSaved results to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'HitRate@1':>10} {'HitRate@5':>10} {'HitRate@20':>11} {'N':>6}")
    print(f"{'-'*60}")
    for label, res in sorted(all_results.items()):
        print(f"{label:<20} {res.get('HitRate@1',0):>10.4f} {res.get('HitRate@5',0):>10.4f} {res.get('HitRate@20',0):>11.4f} {res.get('n_evaluated',0):>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
