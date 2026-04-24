#!/usr/bin/env python3
"""
Generate subformulae assignment JSONs for MIST-CF predicted formulas.

For each test spectrum, reads the top-K predicted formulas from MIST-CF output,
loads the raw MS2 spectrum, computes subformula assignments (matching MS2 peaks
to fragments of the candidate formula), and writes a JSON file per spectrum in
the format expected by MassSpecGym's MsgSubformulaFeaturizer.

Usage:
    python scripts/create_mistcf_subformulae.py \
        --predictions /path/to/top5_predictions.json \
        --spec-dir /path/to/msg/spec_files \
        --labels /path/to/msg/labels.tsv \
        --split /path/to/msg/split.tsv \
        --output-dir /path/to/msg/subformulae/mistcf_top1_subformulae \
        --top-k 1 \
        --num-workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache, reduce
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Spec file parsing (self-contained, mirrors mist_cf.common.parse_utils)
# ---------------------------------------------------------------------------

def parse_spectra(spectra_file: str) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]
    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )
    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                spectras.append((spectra_header, peak_data))
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end
            metadata.update(entries)
        group_num += 1
    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def load_spectrum(spec_name: str, spec_dir: Path, max_peaks: int = 100, inten_thresh: float = 0.0) -> Optional[np.ndarray]:
    spec_file = spec_dir / f"{spec_name}.ms"
    if not spec_file.is_file():
        return None
    meta, tuples = parse_spectra(str(spec_file))
    parent_mass = meta.get("precursor_mz", meta.get("parentmass"))
    if parent_mass is None:
        return None
    parent_mass = float(parent_mass)

    fused = [x for _, x in tuples if x.size > 0]
    if not fused:
        return None

    # Merge duplicate mz values
    mz_to_inten = {}
    mz_to_mz = {}
    for arr in fused:
        for mz, inten in arr:
            key = np.round(mz, 4)
            if key not in mz_to_inten or inten > mz_to_inten[key]:
                mz_to_inten[key] = inten
                mz_to_mz[key] = mz
    pairs = np.array([[mz_to_mz[k], mz_to_inten[k]] for k in mz_to_mz])
    pairs = pairs[pairs[:, 0] <= (parent_mass + 1)]
    if pairs.size == 0:
        return None
    # Normalize
    max_inten = pairs[:, 1].max()
    if max_inten > 0:
        pairs[:, 1] /= max_inten

    # Top peaks + intensity threshold
    order = np.argsort(pairs[:, 1])[::-1][:max_peaks]
    pairs = pairs[order]
    mask = pairs[:, 1] > inten_thresh
    pairs = pairs[mask]
    return pairs if pairs.size > 0 else None


# ---------------------------------------------------------------------------
# Chemistry constants (mirrors mist_cf.common.chem_utils)
# ---------------------------------------------------------------------------

VALID_ELEMENTS = [
    "C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Se", "B", "Si"
]

ELEMENT_TO_MASS = {
    "C": 12.0, "H": 1.00782503207, "N": 14.0030740048,
    "O": 15.99491461956, "P": 30.97376163, "S": 31.97207100,
    "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371,
    "I": 126.904473, "Se": 79.9165213, "B": 11.0093054,
    "Si": 27.9769265325,
}

VALID_MONO_MASSES = np.array([ELEMENT_TO_MASS[e] for e in VALID_ELEMENTS])
NUM_ELEMENTS = len(VALID_ELEMENTS)

# RDBE multipliers for each element: C=2, H=-1, N=1, O=0, P=1, S=0, F=-1, Cl=-1, Br=-1, I=-1, Se=0, B=1, Si=2
RDBE_MULT = np.array([2, -1, 1, 0, 1, 0, -1, -1, -1, -1, 0, 1, 2], dtype=float)

ELEMENT_VECTORS = np.eye(NUM_ELEMENTS, dtype=float)

ION_TO_MASS = {
    "[M+H]+": 1.00727645224,
    "[M-H]-": -1.00727645224,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+Cl]-": 34.969402,
    "[M+NH4]+": 18.034164,
}

INSTRUMENT_TO_TYPE = {
    "Orbitrap": "orbitrap",
    "Q-TOF": "qtof",
    "FT-ICR": "fticr",
}
INSTRUMENT_TO_TOL = {
    "orbitrap": 10,
    "qtof": 15,
    "fticr": 5,
    "unknown": 15,
}


def get_instr_tol(instrument: str) -> int:
    inst = INSTRUMENT_TO_TYPE.get(instrument, "unknown")
    return INSTRUMENT_TO_TOL[inst]


import re
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def formula_to_dense(formula: str) -> np.ndarray:
    vec = np.zeros(NUM_ELEMENTS, dtype=float)
    for elem, count in FORMULA_RE.findall(formula):
        if not elem:
            continue
        count = int(count) if count else 1
        if elem in VALID_ELEMENTS:
            vec[VALID_ELEMENTS.index(elem)] = count
    return vec


def vec_to_formula(vec: np.ndarray) -> str:
    parts = []
    for i, count in enumerate(vec):
        c = int(count)
        if c > 0:
            parts.append(f"{VALID_ELEMENTS[i]}{c if c > 1 else ''}")
    return "".join(parts)


def cross_sum(x, y):
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def rdbe_filter(cross_prod):
    rdbe_total = 1 + 0.5 * cross_prod.dot(RDBE_MULT)
    return np.argwhere(rdbe_total >= 0).flatten()


@lru_cache(maxsize=4096)
def get_all_subsets(chem_formula: str):
    dense = formula_to_dense(chem_formula)
    non_zero = np.argwhere(dense > 0).flatten()
    vectorized = []
    for idx in non_zero:
        temp = ELEMENT_VECTORS[idx] * np.arange(0, dense[idx] + 1).reshape(-1, 1)
        vectorized.append(temp)
    zero_vec = np.zeros((1, NUM_ELEMENTS))
    cross_prod = reduce(cross_sum, vectorized, zero_vec)
    valid = rdbe_filter(cross_prod)
    cross_prod = cross_prod[valid]
    masses = cross_prod.dot(VALID_MONO_MASSES)
    return cross_prod, masses


def clipped_ppm(mass_diff, parentmass):
    pm = parentmass.copy()
    pm[pm < 200] = 200
    return mass_diff / pm * 1e6


def assign_subforms(form: str, spec: np.ndarray, ion_type: str, mass_diff_thresh: float = 15) -> dict:
    """Assign subformulae of `form` to MS2 peaks in `spec`."""
    if ion_type not in ION_TO_MASS:
        return {"cand_form": form, "cand_ion": ion_type, "output_tbl": None}

    try:
        cross_prod, masses = get_all_subsets(form)
    except (MemoryError, np.core._exceptions._ArrayMemoryError):
        print(f"  [WARN] OOM computing subsets for {form}, skipping")
        return {"cand_form": form, "cand_ion": ion_type, "output_tbl": None}

    spec_masses, spec_intens = spec[:, 0], spec[:, 1]
    ion_mass = ION_TO_MASS[ion_type]
    masses_with_ion = masses + ion_mass
    ion_types = np.array([ion_type] * len(masses_with_ion))

    diffs = np.abs(spec_masses[:, None] - masses_with_ion[None, :])
    formula_inds = diffs.argmin(-1)
    min_diff = diffs[np.arange(len(diffs)), formula_inds]
    rel_diff = clipped_ppm(min_diff, spec_masses)

    valid = rel_diff < mass_diff_thresh
    spec_masses = spec_masses[valid]
    spec_intens = spec_intens[valid]
    min_diff = min_diff[valid]
    rel_diff = rel_diff[valid]
    formula_inds = formula_inds[valid]

    formulas = np.array([vec_to_formula(j) for j in cross_prod[formula_inds]])
    formula_masses = masses_with_ion[formula_inds]
    ion_types = ion_types[formula_inds]

    # Deduplicate formulas, summing intensities
    seen = {}
    uniq = []
    for idx, f in enumerate(formulas):
        uniq.append(f not in seen)
        if f in seen:
            spec_intens[seen[f]] += spec_intens[idx]
        seen[f] = idx

    uniq = np.array(uniq)
    if uniq.sum() == 0:
        return {"cand_form": form, "cand_ion": ion_type, "output_tbl": None}

    spec_masses = spec_masses[uniq]
    spec_intens = spec_intens[uniq]
    min_diff = min_diff[uniq]
    rel_diff = rel_diff[uniq]
    formula_masses = formula_masses[uniq]
    formulas = formulas[uniq]
    ion_types = ion_types[uniq]

    output_tbl = {
        "mz": [float(x) for x in spec_masses],
        "ms2_inten": [float(x) for x in spec_intens],
        "mono_mass": [float(x) for x in formula_masses],
        "abs_mass_diff": [float(x) for x in min_diff],
        "mass_diff": [float(x) for x in rel_diff],
        "formula": list(formulas),
        "ions": list(ion_types),
    }
    return {"cand_form": form, "cand_ion": ion_type, "output_tbl": output_tbl}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def process_one(args_tuple):
    """Process a single (spectrum, formula, rank) triple."""
    spec_name, formula, ion_type, spec_dir, mass_tol, output_dir, max_peaks = args_tuple
    out_path = output_dir / f"{spec_name}.json"
    if out_path.exists():
        return spec_name, True

    spec = load_spectrum(spec_name, spec_dir, max_peaks=max_peaks)
    if spec is None:
        result = {"cand_form": formula, "cand_ion": ion_type, "output_tbl": None}
    else:
        result = assign_subforms(formula, spec, ion_type, mass_diff_thresh=mass_tol)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    return spec_name, True


def process_all_ranks(args_tuple):
    """Process all 5 formula ranks for one spectrum in a single call."""
    spec_name, formulas, ion_type, spec_dir, mass_tol, output_dirs, max_peaks = args_tuple

    spec = load_spectrum(spec_name, spec_dir, max_peaks=max_peaks)

    for rank_idx, (formula, out_dir) in enumerate(zip(formulas, output_dirs)):
        out_path = out_dir / f"{spec_name}.json"
        if out_path.exists():
            continue
        if spec is None or formula is None:
            result = {"cand_form": formula or "", "cand_ion": ion_type, "output_tbl": None}
        else:
            result = assign_subforms(formula, spec, ion_type, mass_diff_thresh=mass_tol)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)

    return spec_name, True


def main():
    parser = argparse.ArgumentParser(description="Generate subformulae from MIST-CF predicted formulas")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to top5_predictions.json (spec_id -> [formula1, ..., formula5])")
    parser.add_argument("--spec-dir", type=Path, required=True,
                        help="Directory containing .ms spec files")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Path to labels.tsv")
    parser.add_argument("--split", type=Path, required=True,
                        help="Path to split.tsv")
    parser.add_argument("--output-base", type=Path, required=True,
                        help="Base output dir; creates mistcf_rank{1..K}_subformulae/ inside")
    parser.add_argument("--max-k", type=int, default=5,
                        help="Generate subformulae for ranks 1..max_k (default 5)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-peaks", type=int, default=100)
    parser.add_argument("--test-only", action="store_true", default=True,
                        help="Only generate subformulae for test set")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    max_k = args.max_k
    output_dirs = []
    for k in range(1, max_k + 1):
        d = args.output_base / f"mistcf_rank{k}_subformulae"
        d.mkdir(parents=True, exist_ok=True)
        output_dirs.append(d)
    print(f"Output dirs: {[str(d) for d in output_dirs]}")

    with open(args.predictions) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} spectrum predictions")

    labels = pd.read_csv(args.labels, sep="\t")
    col = "spec" if "spec" in labels.columns else "identifier"
    labels = labels.rename(columns={col: "identifier"})

    split = pd.read_csv(args.split, sep="\t")
    if "name" in split.columns:
        split = split.rename(columns={"name": "identifier", "split": "fold"})
    elif "spec" in split.columns:
        split = split.rename(columns={"spec": "identifier"})

    df = labels.merge(split[["identifier", "fold"]], on="identifier", how="inner")
    if args.test_only:
        df = df[df["fold"] == "test"]
    print(f"Processing {len(df)} spectra (test_only={args.test_only})")

    if args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx or 0
        e = args.end_idx or len(df)
        df = df.iloc[s:e]
        print(f"Sliced to [{s}:{e}] = {len(df)} spectra")

    tasks = []
    missing = 0
    for _, row in df.iterrows():
        spec_name = str(row["identifier"])
        ion_type = str(row.get("ionization", "[M+H]+"))
        instrument = str(row.get("instrument", ""))
        mass_tol = get_instr_tol(instrument)

        if spec_name not in preds:
            missing += 1
            continue

        pred_formulas = preds[spec_name]
        # Pad with None if fewer than max_k predictions
        padded = list(pred_formulas[:max_k]) + [None] * max(0, max_k - len(pred_formulas))

        tasks.append((spec_name, padded, ion_type, args.spec_dir, mass_tol, output_dirs, args.max_peaks))

    if missing > 0:
        print(f"[WARN] {missing} spectra had no MIST-CF prediction, skipped")

    total_assignments = len(tasks) * max_k
    already_done = 0
    for t in tasks:
        for d in output_dirs:
            if (d / f"{t[0]}.json").exists():
                already_done += 1
    print(f"Total assignments: {total_assignments}, already done: {already_done}, remaining: {total_assignments - already_done}")

    if args.num_workers <= 1:
        from tqdm import tqdm
        for t in tqdm(tasks, desc="Assigning subformulae (all ranks)"):
            process_all_ranks(t)
    else:
        from multiprocessing import Pool
        from tqdm import tqdm
        with Pool(args.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_all_ranks, tasks),
                          total=len(tasks), desc="Assigning subformulae (all ranks)"):
                pass

    print("Done!")


if __name__ == "__main__":
    main()
