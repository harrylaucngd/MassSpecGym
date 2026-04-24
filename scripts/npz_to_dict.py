#!/usr/bin/env python3
"""Convert an NPZ of MIST-predicted test fingerprints to a pickled dict for FromDictRetrieval.

Input npz is expected to contain arrays:
    spec_id: (N,) string
    fingerprint: (N, D) numeric
Output: pickle of {spec_id: np.ndarray(D, float32)} suitable for FromDictRetrieval(dct_path=...).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Output pickle path.")
    p.add_argument("--id-key", default="spec_id")
    p.add_argument("--fp-key", default="fingerprint")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    ids = data[args.id_key]
    fps = data[args.fp_key]
    assert len(ids) == len(fps), f"len mismatch: {len(ids)} vs {len(fps)}"
    print(f"Loaded {len(ids)} fps, dtype={fps.dtype}, shape={fps.shape}")

    dct = {str(i): fps[k].astype(np.float32) for k, i in enumerate(ids)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dct, f)
    print(f"Wrote {len(dct)} fps -> {args.out}")


if __name__ == "__main__":
    main()
