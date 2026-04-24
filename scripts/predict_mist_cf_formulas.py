#!/usr/bin/env python3
"""
Run MIST-CF formula prediction on the MassSpecGym dataset and save results.

Produces a JSON file mapping spectrum identifier -> top-predicted formula,
which can be passed as formula_pth to MassSpecDataset(formula_source="inferred").

Usage:
    python scripts/predict_mist_cf_formulas.py \\
        --checkpoint checkpoints/mist_cf_best.ckpt \\
        --tsv data/MassSpecGym.tsv \\
        --output data/mist_cf_predictions.json \\
        --split test \\
        --top-k 1

    # With custom candidate set (skip SIRIUS):
    python scripts/predict_mist_cf_formulas.py \\
        --checkpoint checkpoints/mist_cf_best.ckpt \\
        --tsv data/MassSpecGym.tsv \\
        --candidates-dir data/msg/candidates/ \\
        --output data/mist_cf_predictions.json

    # With fast filter:
    python scripts/predict_mist_cf_formulas.py \\
        --checkpoint checkpoints/mist_cf_best.ckpt \\
        --fast-filter-checkpoint checkpoints/fast_ffn_best.ckpt \\
        --tsv data/MassSpecGym.tsv \\
        --output data/mist_cf_predictions.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to MistCFNet checkpoint")
    p.add_argument("--tsv", default=None, help="Path to MassSpecGym.tsv (downloads if not given)")
    p.add_argument("--output", required=True, help="Output JSON path: {identifier: formula}")
    p.add_argument("--split", default=None, help="Restrict to a data split (e.g. 'test')")
    p.add_argument("--top-k", type=int, default=1, help="Number of top formulas to store per spectrum (default 1)")
    p.add_argument("--fast-filter-checkpoint", default=None, help="Optional FastFFN checkpoint for pre-filtering")
    p.add_argument("--fast-filter-max-k", type=int, default=256, help="Candidates kept after fast filter")
    p.add_argument("--ppm-tol", type=int, default=15, help="SIRIUS PPM tolerance")
    p.add_argument("--el-str", default=None, help="SIRIUS element string (uses default if not set)")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available")
    p.add_argument("--batch-size", type=int, default=64, help="MistCFNet scoring batch size")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    from massspecgym.models.oracles.mist_cf import MistCFNet, FastFFN, predict_formulas

    # Load model
    model = MistCFNet.from_pretrained(args.checkpoint)
    model = model.to(device).eval()
    logger.info(f"Loaded MistCFNet from {args.checkpoint}")

    # Load fast filter
    fast_filter = None
    if args.fast_filter_checkpoint:
        fast_filter = FastFFN.load_from_checkpoint(args.fast_filter_checkpoint)
        fast_filter = fast_filter.to(device).eval()
        logger.info(f"Loaded FastFFN from {args.fast_filter_checkpoint}")

    # Load dataset
    if args.tsv is None:
        import massspecgym.utils as utils
        tsv_path = utils.hugging_face_download("MassSpecGym.tsv")
    else:
        tsv_path = args.tsv

    df = pd.read_csv(tsv_path, sep="\t")
    if args.split is not None:
        df = df[df["fold"] == args.split].reset_index(drop=True)
        logger.info(f"Filtered to split='{args.split}': {len(df)} spectra")

    # Check for existing output to resume
    predictions = {}
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            predictions = json.load(f)
        logger.info(f"Resuming from {out_path}: {len(predictions)} already done")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting formulas"):
        identifier = str(row["identifier"])
        if identifier in predictions:
            continue

        mzs = [float(m) for m in str(row["mzs"]).split(",")]
        intensities = [float(i) for i in str(row["intensities"]).split(",")]
        precursor_mz = float(row["precursor_mz"])
        adduct = str(row.get("adduct", "[M+H]+"))

        try:
            results = predict_formulas(
                spectrum_mzs=mzs,
                spectrum_intensities=intensities,
                precursor_mz=precursor_mz,
                adduct=adduct,
                top_k=args.top_k,
                model=model,
                fast_filter_model=fast_filter,
                fast_filter_max_k=args.fast_filter_max_k,
                ppm_tol=args.ppm_tol,
                el_str=args.el_str,
                device=device,
                batch_size=args.batch_size,
            )
        except Exception as e:
            logger.warning(f"Failed for {identifier}: {e}")
            predictions[identifier] = ""
            continue

        if results:
            if args.top_k == 1:
                predictions[identifier] = results[0].formula
            else:
                predictions[identifier] = [r.formula for r in results]
        else:
            predictions[identifier] = "" if args.top_k == 1 else []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
