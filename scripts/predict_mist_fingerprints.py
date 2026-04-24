#!/usr/bin/env python3
"""
Predict MIST fingerprints from subformulae JSONs for all formula ranks.

Loads encoder_msg.pt, iterates through subformulae folders (rank 1..K),
and saves predicted fingerprints as .pt files (dict: spec_id -> 4096-D binary tensor).

Usage:
    python scripts/predict_mist_fingerprints.py \
        --mist-ckpt checkpoints/encoder_msg.pt \
        --subform-base data/msg/subformulae \
        --labels data/msg/labels.tsv \
        --split data/msg/split.tsv \
        --output-dir results/mist_fps \
        --max-k 1 \
        --also-gt \
        --accelerator gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from massspecgym.models.encoders.mist import SpectraEncoderGrowing
from massspecgym.models.encoders.mist.chem_constants import (
    VALID_ELEMENTS,
    formula_to_dense,
    ion_remap,
    ION_LST,
    get_instr_idx,
)

ion_to_idx: dict[str, int] = {ion: i for i, ion in enumerate(ION_LST)}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_torch_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def threshold_fingerprint(fp_prob: torch.Tensor, threshold: float = 0.187) -> torch.Tensor:
    return (fp_prob > threshold).float()


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    if not all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _normalize_encoder_state_dict(sd: dict) -> dict:
    anchor = "spectra_encoder.0.intermediate_layer.input_layer.weight"
    if anchor in sd:
        return sd
    for pref in ("encoder.", "model.encoder.", "model.module.encoder.", "module.encoder.", "module."):
        if (pref + anchor) in sd:
            return {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    matches = [k for k in sd.keys() if k.endswith(anchor)]
    if not matches:
        raise ValueError(f"Cannot locate encoder weights; anchor key not found: '{anchor}'")
    pref = matches[0][:-len(anchor)]
    return {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}


DEFAULT_MSG_ENCODER_KWARGS: dict[str, T.Any] = {
    "inten_transform": "float",
    "peak_attn_layers": 2,
    "num_heads": 8,
    "pairwise_featurization": True,
    "embed_instrument": False,
    "set_pooling": "cls",
    "form_embedder": "pos-cos",
    "output_size": 4096,
    "hidden_size": 512,
    "spectra_dropout": 0.1,
    "top_layers": 1,
    "refine_layers": 4,
    "magma_modulo": 2048,
    "instr_dim": 6,
}

_N_ELEMENTS = len(VALID_ELEMENTS)
_CLS_TYPE = 3   # matches FormulaTransformer.cls_type in modules.py
_FRAG_TYPE = 0


class MsgSubformulaFeaturizer:
    """Converts a per-spectrum subformulae JSON into the batch dict expected by
    SpectraEncoderGrowing / FormulaTransformer."""

    def __init__(self, subform_folder: Path, cls_mode: str = "ms1", max_peaks: int = 60):
        self.subform_folder = Path(subform_folder)
        self.max_peaks = max_peaks  # max fragment peaks (excl. CLS)

    def featurize_one(self, identifier: str, instrument: str) -> dict[str, torch.Tensor]:
        json_path = self.subform_folder / f"{identifier}.json"
        d = json.load(open(json_path))

        cand_form: str = d["cand_form"]
        cand_ion: str  = d["cand_ion"]
        tbl: dict      = d["output_tbl"]

        # Ion index (unknown → 0)
        ion = ion_remap.get(cand_ion, ION_LST[0])
        ion_idx = ion_to_idx.get(ion, 0)

        # Instrument index
        instr_idx = get_instr_idx(instrument)

        # Fragment peaks
        formulas = tbl["formula"]
        intens_raw = tbl["ms2_inten"]
        n_frags = min(len(formulas), self.max_peaks)

        parent_vec = formula_to_dense(cand_form)                     # (n_el,)
        frag_vecs  = np.zeros((n_frags, _N_ELEMENTS), dtype=np.int64)
        frag_intens = np.zeros(n_frags, dtype=np.float32)
        for i in range(n_frags):
            try:
                frag_vecs[i] = formula_to_dense(formulas[i])
            except Exception:
                pass
            frag_intens[i] = float(intens_raw[i])

        # Stack: [CLS, frag1, frag2, ...]
        n_total = 1 + n_frags
        form_vecs = np.vstack([parent_vec[None], frag_vecs])         # (n_total, n_el)
        intens    = np.concatenate([[1.0], frag_intens])             # (n_total,)
        types     = np.array([_CLS_TYPE] + [_FRAG_TYPE] * n_frags, dtype=np.int64)

        return {
            "num_peaks":   torch.tensor([n_total],                          dtype=torch.long),
            "types":       torch.tensor(types,          dtype=torch.long).unsqueeze(0),   # (1, N)
            "instruments": torch.tensor([instr_idx],                        dtype=torch.long),
            "ion_vec":     torch.full((1, n_total), ion_idx,                dtype=torch.long),  # (1, N)
            "form_vec":    torch.tensor(form_vecs,      dtype=torch.long).unsqueeze(0),   # (1, N, n_el)
            "intens":      torch.tensor(intens,         dtype=torch.float32).unsqueeze(0),  # (1, N)
        }


# ── model loading ──────────────────────────────────────────────────────────────

def load_encoder(ckpt_path: str, device: torch.device) -> SpectraEncoderGrowing:
    sd = load_torch_state_dict(ckpt_path)
    sd = _strip_prefix_if_present(sd, "model.")
    sd = _strip_prefix_if_present(sd, "encoder.")
    sd = _normalize_encoder_state_dict(sd)

    encoder = SpectraEncoderGrowing(**DEFAULT_MSG_ENCODER_KWARGS)
    model_sd = encoder.state_dict()
    filtered = {
        k: v for k, v in sd.items()
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape
    }
    encoder.load_state_dict(filtered, strict=False)
    encoder.eval()
    encoder.to(device)
    return encoder


@torch.no_grad()
def predict_fingerprint(
    encoder: SpectraEncoderGrowing,
    mist_input: dict[str, torch.Tensor],
    device: torch.device,
    threshold: float = 0.187,
    save_raw: bool = False,
) -> torch.Tensor:
    inp = {k: v.to(device=device, non_blocking=True) for k, v in mist_input.items()}
    fp_prob, _ = encoder(inp)
    if save_raw:
        return fp_prob.squeeze(0).cpu()
    return threshold_fingerprint(fp_prob, threshold=threshold).squeeze(0).cpu()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict MIST fingerprints from subformulae")
    parser.add_argument("--mist-ckpt", type=Path, required=True)
    parser.add_argument("--subform-base", type=Path, required=True,
                        help="Base dir containing mistcf_rank{1..K}_subformulae/ and default_subformulae/ folders")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.187)
    parser.add_argument("--save-raw", action="store_true",
                        help="Save raw sigmoid probabilities instead of thresholded binary fps.")
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--also-gt", action="store_true",
                        help="Also predict FPs using default_subformulae (ground-truth formula)")
    parser.add_argument("--gt-subform-folder", type=Path, default=None,
                        help="Override path to GT subformulae folder")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.accelerator in ("gpu", "cuda") and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    encoder = load_encoder(str(args.mist_ckpt), device)
    print(f"Loaded encoder from {args.mist_ckpt}")

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
    print(f"Test set: {len(df)} spectra")

    rank_jobs: list[tuple[str, Path]] = []
    for k in range(1, args.max_k + 1):
        folder = args.subform_base / f"mistcf_rank{k}_subformulae"
        if folder.is_dir():
            rank_jobs.append((f"mistcf_rank{k}", folder))
        else:
            print(f"[WARN] Missing folder: {folder}")

    if args.also_gt:
        gt_folder = args.gt_subform_folder or (args.subform_base / "default_subformulae")
        if gt_folder.is_dir():
            rank_jobs.append(("gt", gt_folder))
        else:
            print(f"[WARN] Missing GT folder: {gt_folder}")

    for rank_label, subform_folder in rank_jobs:
        suffix = "_raw" if args.save_raw else ""
        out_path = args.output_dir / f"fingerprints_{rank_label}{suffix}.pt"
        if out_path.exists():
            print(f"[SKIP] {rank_label}: {out_path} already exists")
            continue

        print(f"\n{'='*60}\nProcessing {rank_label}: {subform_folder}\n{'='*60}")
        featurizer = MsgSubformulaFeaturizer(subform_folder)
        fps: dict[str, torch.Tensor] = {}
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=rank_label):
            spec_id   = str(row["identifier"])
            instrument = str(row.get("instrument", ""))
            if not (subform_folder / f"{spec_id}.json").exists():
                skipped += 1
                continue
            try:
                mist_input = featurizer.featurize_one(spec_id, instrument)
                fps[spec_id] = predict_fingerprint(
                    encoder, mist_input, device,
                    threshold=args.threshold, save_raw=args.save_raw,
                )
            except Exception as e:
                print(f"  [ERR] {spec_id}: {e}")
                skipped += 1

        print(f"  Predicted: {len(fps)}, Skipped: {skipped}")
        torch.save(fps, out_path)
        print(f"  Saved: {out_path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
