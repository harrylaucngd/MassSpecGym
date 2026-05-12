#!/usr/bin/env python3
"""
run_retrieval.py - MIST-based retrieval for MassSpecGym.

Subcommands:
  train       Train/test MIST fingerprint retrieval end-to-end (Lightning).
  predict     Run pretrained MIST encoder over test spectra → save .pt FP files.
  eval        Evaluate precomputed FPs vs candidates (HitRate@K + 95% CI).
  formula     Predict top-K formulas using MistCFNet.
  subformulae Generate per-spectrum subformulae JSONs from MIST-CF formula predictions.
  build-cache Pre-compute Morgan FP cache (.npz) for a candidates JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# train
# ============================================================

def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labels", type=str, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--candidates", type=str, default=None)
    p.add_argument("--subform-folder", type=str, default=None,
                   help="Subformulae JSON folder (e.g. data/subformulae/default_subformulae)")
    p.add_argument("--encoder-checkpoint", type=str, default=None)
    p.add_argument("--fp-save-path", type=str, default=None)
    p.add_argument("--fp-similarity", default="cosine", choices=["cosine", "tanimoto"])
    p.add_argument("--fp-size", type=int, default=4096)
    p.add_argument("--inferred-formula", action="store_true")
    p.add_argument("--inferred-formula-pth", type=str, default=None)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--checkpoint-pth", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--accelerator", type=str, default="gpu")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--job-key", type=str, required=True)
    p.add_argument("--project-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default="mass-spec-ml")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--skip-mces-test", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log-every-n-steps", type=int, default=50)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--df-test-pth", type=Path, default=None)


def cmd_train(args: argparse.Namespace) -> None:
    import datetime
    import typing as T

    import pandas as pd
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from rdkit import RDLogger

    import massspecgym.utils as utils
    from massspecgym.data import MassSpecDataModule
    from massspecgym.models.base import Stage
    from massspecgym.models.retrieval import MISTFingerprintRetrieval
    from massspecgym.definitions import MASSSPECGYM_TEST_RESULTS_DIR

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    pl.seed_everything(args.seed)

    if args.debug:
        args.labels = "../data/debug/example_5_spectra_inferred_formula.json"
        args.candidates = "../data/debug/example_5_spectra_candidates.json"
        args.split = "../data/debug/example_5_spectra_split.tsv"

    # Detect split format
    split_pth = args.split
    mist_split_pth = datamodule_split_pth = None
    if split_pth:
        cols = set(pd.read_csv(split_pth, sep="\t", nrows=0).columns)
        if cols == {"name", "split"}:
            mist_split_pth = split_pth
        elif cols == {"identifier", "fold"}:
            datamodule_split_pth = split_pth
        else:
            raise ValueError(f"Split TSV must have columns (name, split) or (identifier, fold); got {sorted(cols)}")

    # Load test identifiers for --test-only
    identifiers_subset: T.Optional[T.List[str]] = None
    if args.test_only:
        if datamodule_split_pth:
            df = pd.read_csv(datamodule_split_pth, sep="\t")
            identifiers_subset = df.loc[df["fold"] == "test", "identifier"].astype(str).tolist()
        elif mist_split_pth:
            df = pd.read_csv(mist_split_pth, sep="\t")
            df = df.rename(columns={"name": "identifier", "split": "fold"})
            identifiers_subset = df.loc[df["fold"] == "test", "identifier"].astype(str).tolist()

    from massspecgym.data.mist_dataset import MISTRetrievalDataset
    dataset = MISTRetrievalDataset(
        subform_folder=args.subform_folder,
        mist_split_pth=mist_split_pth,
        pth=args.labels,
        fp_size=args.fp_size,
        candidates_pth=args.candidates,
        inferred_formula=args.inferred_formula,
        inferred_formula_pth=args.inferred_formula_pth,
        identifiers_subset=identifiers_subset,
    )

    data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=None if args.test_only else datamodule_split_pth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    now_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formula_tag = "_inferred_formula" if args.inferred_formula else ""
    if args.df_test_pth is None and args.devices == 1:
        args.df_test_pth = MASSSPECGYM_TEST_RESULTS_DIR / f"retrieval/{args.run_name}{formula_tag}_{now_tag}.pkl"

    no_mces = [Stage.VAL] + ([Stage.TEST] if args.skip_mces_test else [])
    common_kw = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        no_mces_metrics_at_stages=no_mces,
        df_test_path=args.df_test_pth,
    )

    model = MISTFingerprintRetrieval(
        encoder_checkpoint=args.encoder_checkpoint,
        fp_bits=args.fp_size,
        similarity=args.fp_similarity,
        fp_save_path=args.fp_save_path,
        **common_kw,
    )

    if args.checkpoint_pth is not None:
        model = MISTFingerprintRetrieval.load_from_checkpoint(
            args.checkpoint_pth,
            no_mces_metrics_at_stages=no_mces,
            df_test_path=args.df_test_pth,
        )

    logger = None
    if not args.no_wandb:
        logger = pl.loggers.WandbLogger(
            name=args.run_name,
            project=args.project_name or "MassSpecGymRetrieval",
            entity=args.wandb_entity,
            log_model=False,
            config=vars(args),
        )

    callbacks = []
    for i, monitor in enumerate(model.get_checkpoint_monitors()):
        callbacks.append(pl.callbacks.ModelCheckpoint(
            monitor=monitor["monitor"],
            save_top_k=1,
            mode=monitor["mode"],
            dirpath=Path(args.project_name or "checkpoints") / args.job_key,
            filename=f"{{step:06d}}-{{{monitor['monitor']}:03.03f}}",
            auto_insert_metric_name=True,
            save_last=(i == 0),
        ))
        if monitor.get("early_stopping", False):
            callbacks.append(EarlyStopping(monitor=monitor["monitor"], mode=monitor["mode"], verbose=True))

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
    )

    data_module.prepare_data()
    data_module.setup()

    if not args.test_only:
        trainer.validate(model, datamodule=data_module)
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


# ============================================================
# predict
# ============================================================

def add_predict_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mist-ckpt", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--formula-predictions", type=Path, default=None,
                   help="JSON: {spec_id: [formula_rank1, ..., formula_rankK]} from any method")
    p.add_argument("--subform-cache", type=Path, default=None,
                   help="Dir for subformulae JSONs (default: output-dir/subformulae/)")
    p.add_argument("--spec-dir", type=Path, default=None,
                   help=".ms files dir; required when subformulae need to be generated on-the-fly")
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.187)
    p.add_argument("--save-raw", action="store_true",
                   help="Save raw sigmoid probabilities instead of thresholded binary fps")
    p.add_argument("--also-gt", action="store_true",
                   help="Also run encoder with ground-truth formula (rank0) if present in subform-cache")
    p.add_argument("--ion-type", type=str, default="[M+H]+",
                   help="Ion type for on-the-fly subformulae generation")
    p.add_argument("--mass-tol", type=float, default=15.0,
                   help="PPM tolerance for subformulae assignment")
    p.add_argument("--accelerator", type=str, default="cpu")


def _load_encoder_state_dict(path: str) -> dict:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("state_dict", obj) if isinstance(obj, dict) else obj


def _normalize_encoder_sd(sd: dict) -> dict:
    anchor = "spectra_encoder.0.intermediate_layer.input_layer.weight"
    if anchor in sd:
        return sd
    for pref in ("encoder.", "model.encoder.", "model.module.encoder.", "module.encoder.", "module."):
        if (pref + anchor) in sd:
            return {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    matches = [k for k in sd if k.endswith(anchor)]
    if not matches:
        raise ValueError(f"Cannot locate encoder weights; anchor key '{anchor}' not found")
    pref = matches[0][: -len(anchor)]
    return {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}


class MsgSubformulaFeaturizer:
    """Convert per-spectrum subformulae JSON → batch dict for SpectraEncoderGrowing."""

    _CLS_TYPE = 3
    _FRAG_TYPE = 0

    def __init__(self, subform_folder: Path, max_peaks: int = 60):
        from massspecgym.models.encoders.mist.chem_constants import (
            VALID_ELEMENTS, ion_remap, ION_LST, get_instr_idx,
        )
        self.subform_folder = Path(subform_folder)
        self.max_peaks = max_peaks
        self._n_elements = len(VALID_ELEMENTS)
        self._ion_to_idx = {ion: i for i, ion in enumerate(ION_LST)}
        self._ion_remap = ion_remap
        self._ion_lst = ION_LST
        self._get_instr_idx = get_instr_idx
        from massspecgym.models.encoders.mist.chem_constants import formula_to_dense
        self._formula_to_dense = formula_to_dense

    def featurize_one(self, identifier: str, instrument: str) -> dict:
        import numpy as np
        import torch

        d = json.load(open(self.subform_folder / f"{identifier}.json"))
        cand_form: str = d["cand_form"]
        cand_ion: str = d["cand_ion"]
        tbl: dict = d["output_tbl"]

        ion = self._ion_remap.get(cand_ion, self._ion_lst[0])
        ion_idx = self._ion_to_idx.get(ion, 0)
        instr_idx = self._get_instr_idx(instrument)

        formulas = tbl["formula"]
        intens_raw = tbl["ms2_inten"]
        n_frags = min(len(formulas), self.max_peaks)

        parent_vec = self._formula_to_dense(cand_form)
        frag_vecs = np.zeros((n_frags, self._n_elements), dtype=np.int64)
        frag_intens = np.zeros(n_frags, dtype=np.float32)
        for i in range(n_frags):
            try:
                frag_vecs[i] = self._formula_to_dense(formulas[i])
            except Exception:
                pass
            frag_intens[i] = float(intens_raw[i])

        n_total = 1 + n_frags
        form_vecs = np.vstack([parent_vec[None], frag_vecs])
        intens = np.concatenate([[1.0], frag_intens])
        types = np.array([self._CLS_TYPE] + [self._FRAG_TYPE] * n_frags, dtype=np.int64)

        return {
            "num_peaks":   torch.tensor([n_total], dtype=torch.long),
            "types":       torch.tensor(types, dtype=torch.long).unsqueeze(0),
            "instruments": torch.tensor([instr_idx], dtype=torch.long),
            "ion_vec":     torch.full((1, n_total), ion_idx, dtype=torch.long),
            "form_vec":    torch.tensor(form_vecs, dtype=torch.long).unsqueeze(0),
            "intens":      torch.tensor(intens, dtype=torch.float32).unsqueeze(0),
        }


def _load_mist_encoder(ckpt_path: str, device):
    import torch
    from massspecgym.models.encoders.mist import SpectraEncoderGrowing

    sd = _load_encoder_state_dict(ckpt_path)
    sd = _normalize_encoder_sd(sd)

    hidden_size = sd["spectra_encoder.0.intermediate_layer.input_layer.bias"].shape[0]
    magma_modulo = sd["spectra_encoder.1.0.weight"].shape[0]

    enc_kw = dict(
        form_embedder="pos-cos", spectra_dropout=0.1, inten_transform="float",
        embed_instrument=False, set_pooling="cls", peak_attn_layers=2,
        num_heads=8, refine_layers=4, pairwise_featurization=True, instr_dim=6,
        output_size=4096, hidden_size=hidden_size, magma_modulo=magma_modulo,
    )
    encoder = SpectraEncoderGrowing(**enc_kw)
    model_sd = encoder.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape}
    n_model = len(model_sd)
    if len(filtered) < n_model // 2:
        raise RuntimeError(f"Only {len(filtered)}/{n_model} weights loaded — architecture mismatch?")
    log.info(f"Loaded {len(filtered)}/{n_model} weights from {ckpt_path} (hidden_size={hidden_size})")
    encoder.load_state_dict(filtered, strict=False)
    encoder.eval().to(device)
    return encoder


def _ensure_subform_json(
    spec_id: str,
    formula: str,
    ion_type: str,
    subform_cache: Path,
    spec_dir: Path | None,
    mass_tol: float,
) -> Path | None:
    """Return path to subformulae JSON, generating it on-the-fly if needed."""
    out_path = subform_cache / f"{spec_id}.json"
    if out_path.exists():
        return out_path
    if spec_dir is None:
        return None
    from massspecgym.preprocessing.subformulae import process_one
    subform_cache.mkdir(parents=True, exist_ok=True)
    process_one((spec_id, formula, ion_type, spec_dir, mass_tol, subform_cache, 100))
    return out_path if out_path.exists() else None


def cmd_predict(args: argparse.Namespace) -> None:
    import torch
    import pandas as pd
    from tqdm import tqdm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subform_cache = args.subform_cache or (args.output_dir / "subformulae")
    device = torch.device("cuda" if args.accelerator in ("gpu", "cuda") and torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    encoder = _load_mist_encoder(str(args.mist_ckpt), device)

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
    log.info(f"Test set: {len(df)} spectra")

    # Build list of (rank_label, {spec_id: formula}) jobs
    rank_jobs: list[tuple[str, dict | None]] = []

    if args.formula_predictions is not None:
        with open(args.formula_predictions) as f:
            formula_preds: dict[str, list[str]] = json.load(f)
        for k in range(1, args.max_k + 1):
            rank_map = {}
            for spec_id, forms in formula_preds.items():
                if len(forms) >= k:
                    rank_map[spec_id] = forms[k - 1]
            if rank_map:
                rank_jobs.append((f"rank{k}", rank_map))
            else:
                log.warning(f"No formulas at rank {k}, stopping.")
                break
    else:
        # No formula predictions — use whatever is already in subform_cache
        rank_jobs.append(("rank1", None))

    if args.also_gt:
        gt_cache = subform_cache / "gt"
        if gt_cache.is_dir() or args.spec_dir is not None:
            rank_jobs.append(("gt", None))
        else:
            log.warning("--also-gt: no gt subformulae found and no --spec-dir given, skipping.")

    for rank_label, formula_map in rank_jobs:
        suffix = "_raw" if args.save_raw else ""
        out_path = args.output_dir / f"fingerprints_{rank_label}{suffix}.pt"
        if out_path.exists():
            log.info(f"[SKIP] {rank_label}: {out_path} already exists")
            continue

        rank_cache = subform_cache / rank_label if formula_map is not None else (
            subform_cache / "gt" if rank_label == "gt" else subform_cache
        )
        rank_cache.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*60}\nProcessing {rank_label}: {rank_cache}\n{'='*60}")
        featurizer = MsgSubformulaFeaturizer(rank_cache)
        fps: dict[str, torch.Tensor] = {}
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=rank_label):
            spec_id = str(row["identifier"])
            instrument = str(row.get("instrument", ""))

            # Resolve subformulae JSON path
            if formula_map is not None:
                formula = formula_map.get(spec_id)
                if formula is None:
                    skipped += 1
                    continue
                json_path = _ensure_subform_json(
                    spec_id, formula, args.ion_type, rank_cache,
                    args.spec_dir, args.mass_tol,
                )
            else:
                json_path = rank_cache / f"{spec_id}.json"
                if not json_path.exists():
                    skipped += 1
                    continue

            if json_path is None or not json_path.exists():
                skipped += 1
                continue

            try:
                mist_input = featurizer.featurize_one(spec_id, instrument)
                inp = {k: v.to(device) for k, v in mist_input.items()}
                with torch.no_grad():
                    fp_prob, _ = encoder(inp)
                fp = fp_prob.squeeze(0).cpu()
                fps[spec_id] = fp if args.save_raw else (fp > args.threshold).float()
            except Exception as e:
                log.warning(f"  {spec_id}: {e}")
                skipped += 1

        log.info(f"  Predicted: {len(fps)}, Skipped: {skipped}")
        torch.save(fps, out_path)
        log.info(f"  Saved: {out_path}")

    log.info("Done.")


# ============================================================
# eval
# ============================================================

def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fp-dir", type=Path, required=True,
                   help="Dir containing fingerprints_rank{k}.pt or fingerprints_mistcf_rank{k}.pt (and optionally fingerprints_gt.pt)")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True,
                   help="MassSpecGym_retrieval_candidates_*.json")
    p.add_argument("--modes", nargs="+",
                   default=["top1", "top5_avg", "top5_max", "top5_rank_avg"],
                   choices=["top1", "top5_avg", "top5_max", "top5_rank_avg"])
    p.add_argument("--max-ranks", type=int, default=5)
    p.add_argument("--hits", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--also-gt", action="store_true")
    p.add_argument("--gt-only", action="store_true")
    p.add_argument("--split-fold", type=str, default="test")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Where to save CI results JSON (default: <fp-dir>/../retrieval_results_<tag>_ci.json)")
    p.add_argument("--fp-cache", type=Path, default=None,
                   help="Optional .npz Morgan FP cache to avoid recomputing RDKit fps")


def _mol_to_fp(smiles: str, fp_size: int = 4096) -> "np.ndarray | None":
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
    import numpy as np
    return np.frombuffer(bv.ToBitString().encode("ascii"), dtype=np.uint8) != ord("0")


def _smiles_to_inchikey(smiles: str) -> "str | None":
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    return Chem.InchiToInchiKey(inchi) if inchi else None


def _tanimoto_batch(query: "np.ndarray", cand_matrix: "np.ndarray") -> "np.ndarray":
    inter = cand_matrix @ query
    union = query.sum() + cand_matrix.sum(axis=1) - inter
    import numpy as np
    return inter / np.maximum(union, 1e-8)


def _scores_to_ranks(scores: "np.ndarray") -> "np.ndarray":
    import numpy as np
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _evaluate_mode(
    mode: str,
    rank_fps: dict,
    df: "pd.DataFrame",
    spec2smiles: dict,
    candidates_json: dict,
    fp_cache: dict,
    ik_cache: dict,
    hit_ks: list,
    max_ranks: int,
) -> dict:
    import numpy as np
    from tqdm import tqdm

    avail = sorted(k for k in rank_fps if k <= max_ranks)

    per_spec_hits = {k: [] for k in hit_ks}
    tanimoto_top1: list = []
    true_ranks: list = []
    norm_ranks: list = []
    n_eval = n_skip_fp = n_skip_cands = 0

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = _mol_to_fp(smi)
        return fp_cache[smi]

    def get_ik(smi):
        if smi not in ik_cache:
            ik_cache[smi] = _smiles_to_inchikey(smi)
        return ik_cache[smi]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=mode):
        spec_id = str(row["identifier"])
        true_smiles = spec2smiles[spec_id]

        if mode == "top1":
            if 1 not in rank_fps or spec_id not in rank_fps[1]:
                n_skip_fp += 1; continue
            fps_for_spec = [rank_fps[1][spec_id]]
        else:
            fps_for_spec = [rank_fps[k][spec_id] for k in avail if spec_id in rank_fps[k]]
            if not fps_for_spec:
                n_skip_fp += 1; continue

        if true_smiles not in candidates_json:
            n_skip_cands += 1; continue
        cand_smiles = candidates_json[true_smiles]

        cand_fps, cand_iks = [], []
        for smi in cand_smiles:
            fp = get_fp(smi); ik = get_ik(smi)
            if fp is not None and ik is not None:
                cand_fps.append(fp); cand_iks.append(ik)
        if not cand_fps:
            n_skip_cands += 1; continue
        true_ik = get_ik(true_smiles)
        if true_ik is None:
            n_skip_cands += 1; continue

        cand_matrix = np.stack(cand_fps).astype(np.float32)

        if mode in ("top1", "top5_avg", "top5_max"):
            if mode == "top1":
                query = fps_for_spec[0].astype(np.float32)
            elif mode == "top5_avg":
                query = np.stack(fps_for_spec).astype(np.float32).mean(axis=0)
            else:
                query = np.stack(fps_for_spec).astype(np.float32).max(axis=0)
            scores = _tanimoto_batch(query, cand_matrix)
            order = np.argsort(scores)[::-1]
        else:  # top5_rank_avg
            rank_mat = np.stack(
                [_scores_to_ranks(_tanimoto_batch(fp.astype(np.float32), cand_matrix))
                 for fp in fps_for_spec]
            ).astype(np.float32)
            avg_ranks = rank_mat.mean(axis=0)
            order = np.argsort(avg_ranks)
            scores = -avg_ranks

        ranked_iks = [cand_iks[i] for i in order]
        try:
            r = ranked_iks.index(true_ik) + 1
        except ValueError:
            r = len(ranked_iks) + 1
        true_ranks.append(r)
        norm_ranks.append(r / len(ranked_iks))

        top1_fp = fps_for_spec[0].astype(np.float32)
        top1_cand = cand_fps[order[0]].astype(np.float32)
        inter = float(np.dot(top1_fp, top1_cand))
        union = float(top1_fp.sum() + top1_cand.sum() - inter)
        tanimoto_top1.append(inter / max(union, 1e-8))

        for k in hit_ks:
            per_spec_hits[k].append(1 if true_ik in ranked_iks[:k] else 0)
        n_eval += 1

    rng = np.random.default_rng(0)
    n_boot = 20000
    res = {
        "mode": mode,
        "n_fps_used": 1 if mode == "top1" else len(avail),
        "n_evaluated": n_eval,
        "skipped_no_fp": n_skip_fp,
        "skipped_no_cands": n_skip_cands,
    }
    for k in hit_ks:
        arr = np.asarray(per_spec_hits[k], dtype=np.float32)
        res[f"HitRate@{k}"] = float(arr.mean()) if arr.size else 0.0
        if arr.size:
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boot = arr[idx].mean(axis=1)
            res[f"HitRate@{k}_lo95"] = float(np.percentile(boot, 2.5))
            res[f"HitRate@{k}_hi95"] = float(np.percentile(boot, 97.5))
        else:
            res[f"HitRate@{k}_lo95"] = res[f"HitRate@{k}_hi95"] = 0.0

    def _ci(arr_list, key):
        arr = np.asarray(arr_list, dtype=np.float32)
        res[key] = float(arr.mean()) if arr.size else 0.0
        if arr.size:
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boot = arr[idx].mean(axis=1)
            res[f"{key}_lo95"] = float(np.percentile(boot, 2.5))
            res[f"{key}_hi95"] = float(np.percentile(boot, 97.5))
        else:
            res[f"{key}_lo95"] = res[f"{key}_hi95"] = 0.0

    _ci(tanimoto_top1, "mean_tanimoto@1")
    _ci(true_ranks, "mean_rank")
    _ci(norm_ranks, "mrr")
    return res


def cmd_eval(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import torch

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    if args.gt_only:
        args.also_gt = True

    # Load split
    labels = pd.read_csv(args.labels, sep="\t")
    split = pd.read_csv(args.split, sep="\t")
    spec_col = "spec" if "spec" in labels.columns else "identifier"
    labels = labels.rename(columns={spec_col: "identifier"})
    s_col = "spec" if "spec" in split.columns else ("name" if "name" in split.columns else "identifier")
    f_col = "fold" if "fold" in split.columns else "split"
    split = split.rename(columns={s_col: "identifier", f_col: "fold"})
    df = labels.merge(split[["identifier", "fold"]], on="identifier", how="inner")
    df = df[df["fold"] == args.split_fold].reset_index(drop=True)
    log.info(f"Test set ({args.split_fold}): {len(df)} spectra")
    spec2smiles = dict(zip(df["identifier"].astype(str), df["smiles"].astype(str)))

    log.info(f"Loading candidates from {args.candidates} ...")
    with open(args.candidates) as f:
        candidates_json: dict = json.load(f)
    log.info(f"  {len(candidates_json)} query SMILES in candidate set")

    # Load FP cache if provided
    fp_cache: dict = {}
    ik_cache: dict = {}
    if args.fp_cache and args.fp_cache.exists():
        data = np.load(args.fp_cache, allow_pickle=True)
        smis = data["smiles"].tolist()
        fps = np.unpackbits(data["fps"], axis=1)[:, :4096].astype(np.float32)
        fp_cache = {s: fps[i] for i, s in enumerate(smis)}
        if "inchikeys" in data.files:
            iks = data["inchikeys"].tolist()
            ik_cache = {s: (ik or None) for s, ik in zip(smis, iks)}
        log.info(f"Loaded {len(fp_cache)} cached fps from {args.fp_cache}")

    # Load rank fingerprints (try new naming first, fall back to old mistcf_ prefix)
    rank_fps: dict[int, dict[str, np.ndarray]] = {}
    if not args.gt_only:
        for k in range(1, args.max_ranks + 1):
            for name in (f"fingerprints_rank{k}.pt", f"fingerprints_mistcf_rank{k}.pt"):
                pth = args.fp_dir / name
                if pth.exists():
                    raw = torch.load(pth, map_location="cpu", weights_only=False)
                    rank_fps[k] = {sid: t.numpy() for sid, t in raw.items()}
                    log.info(f"  Loaded rank {k}: {len(rank_fps[k])} spectra")
                    break
            else:
                log.warning(f"  Missing rank {k} fp file in {args.fp_dir}")

    gt_fps: dict | None = None
    if args.also_gt:
        for name in ("fingerprints_gt.pt", "fingerprints_mistcf_gt.pt"):
            pth = args.fp_dir / name
            if pth.exists():
                raw = torch.load(pth, map_location="cpu", weights_only=False)
                gt_fps = {sid: t.numpy() for sid, t in raw.items()}
                log.info(f"  Loaded GT fingerprints: {len(gt_fps)} spectra")
                break
        if gt_fps is None:
            log.warning(f"--also-gt: no GT fingerprint file found in {args.fp_dir}")

    modes_to_run = [] if args.gt_only else list(args.modes)
    if args.also_gt and gt_fps is not None:
        modes_to_run = ["gt_top1"] + modes_to_run

    all_results: list[dict] = []
    for mode in modes_to_run:
        print(f"\n{'='*60}\n{mode}\n{'='*60}")
        fps_in = {1: gt_fps} if mode == "gt_top1" else rank_fps
        max_r = 1 if mode == "gt_top1" else args.max_ranks
        eval_mode = "top1" if mode == "gt_top1" else mode
        res = _evaluate_mode(eval_mode, fps_in, df, spec2smiles, candidates_json,
                             fp_cache, ik_cache, args.hits, max_r)
        res["mode"] = mode
        all_results.append(res)

        print(f"  Evaluated: {res['n_evaluated']}  "
              f"skipped (no fp): {res['skipped_no_fp']}  "
              f"skipped (no cands): {res['skipped_no_cands']}")
        for k in args.hits:
            print(f"  HitRate@{k}: {res[f'HitRate@{k}']:.4f}  "
                  f"[95% CI {res[f'HitRate@{k}_lo95']:.4f}, {res[f'HitRate@{k}_hi95']:.4f}]")
        print(f"  Mean Tanimoto@1: {res['mean_tanimoto@1']:.4f}")
        print(f"  Mean Rank: {res.get('mean_rank', 0):.1f}")
        print(f"  MRR: {res.get('mrr', 0):.4f}")

    # Summary table
    import pandas as pd
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    df_res = pd.DataFrame(all_results)
    hit_cols = [f"HitRate@{k}" for k in args.hits] + ["mean_tanimoto@1"]
    print(df_res[["mode", "n_fps_used", "n_evaluated"] + hit_cols].to_string(
        index=False, float_format="{:.4f}".format))

    # Save JSON
    cands_tag = args.candidates.stem.split("candidates_")[-1] \
        if "candidates_" in args.candidates.stem else args.candidates.stem
    out_json = args.output_json or (args.fp_dir.parent / f"retrieval_results_{cands_tag}_ci.json")
    json_out = {}
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "mode"}
        json_out[r["mode"]] = entry
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(json_out, indent=2))
    log.info(f"Saved CI results to {out_json}")


# ============================================================
# formula
# ============================================================

def add_formula_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", required=True, help="Path to MistCFNet checkpoint")
    p.add_argument("--labels", type=str, default=None,
                   help="Path to MassSpecGym.tsv (downloads if not given)")
    p.add_argument("--split", type=str, default=None, help="Restrict to split (e.g. 'test')")
    p.add_argument("--output", required=True, help="Output JSON: {identifier: formula}")
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--fast-filter-checkpoint", default=None)
    p.add_argument("--fast-filter-max-k", type=int, default=256)
    p.add_argument("--ppm-tol", type=int, default=15)
    p.add_argument("--el-str", default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)


def cmd_formula(args: argparse.Namespace) -> None:
    import torch
    import pandas as pd
    from tqdm import tqdm

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    from massspecgym.models.oracles.mist_cf import MistCFNet, FastFFN, predict_formulas

    model = MistCFNet.from_pretrained(args.checkpoint).to(device).eval()
    log.info(f"Loaded MistCFNet from {args.checkpoint}")

    fast_filter = None
    if args.fast_filter_checkpoint:
        fast_filter = FastFFN.load_from_checkpoint(args.fast_filter_checkpoint).to(device).eval()
        log.info(f"Loaded FastFFN from {args.fast_filter_checkpoint}")

    if args.labels is None:
        import massspecgym.utils as utils
        tsv_path = utils.hugging_face_download("MassSpecGym.tsv")
    else:
        tsv_path = args.labels

    df = pd.read_csv(tsv_path, sep="\t")
    if args.split is not None:
        df = df[df["fold"] == args.split].reset_index(drop=True)
        log.info(f"Filtered to split='{args.split}': {len(df)} spectra")

    out_path = Path(args.output)
    predictions = {}
    if out_path.exists():
        with open(out_path) as f:
            predictions = json.load(f)
        log.info(f"Resuming from {out_path}: {len(predictions)} already done")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="formula"):
        identifier = str(row["identifier"])
        if identifier in predictions:
            continue
        mzs = [float(m) for m in str(row["mzs"]).split(",")]
        intensities = [float(i) for i in str(row["intensities"]).split(",")]
        precursor_mz = float(row["precursor_mz"])
        adduct = str(row.get("adduct", "[M+H]+"))
        try:
            results = predict_formulas(
                spectrum_mzs=mzs, spectrum_intensities=intensities,
                precursor_mz=precursor_mz, adduct=adduct, top_k=args.top_k,
                model=model, fast_filter_model=fast_filter,
                fast_filter_max_k=args.fast_filter_max_k,
                ppm_tol=args.ppm_tol, el_str=args.el_str,
                device=device, batch_size=args.batch_size,
            )
        except Exception as e:
            log.warning(f"Failed for {identifier}: {e}")
            predictions[identifier] = "" if args.top_k == 1 else []
            continue
        if results:
            predictions[identifier] = results[0].formula if args.top_k == 1 else [r.formula for r in results]
        else:
            predictions[identifier] = "" if args.top_k == 1 else []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    log.info(f"Saved {len(predictions)} predictions to {out_path}")


# ============================================================
# subformulae
# ============================================================

def add_subformulae_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--predictions", type=Path, required=True,
                   help="top-K predictions JSON (spec_id -> [formula1, ..., formulaK])")
    p.add_argument("--spec-dir", type=Path, required=True,
                   help="Directory containing .ms spectrum files")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output-base", type=Path, required=True,
                   help="Base output dir; creates mistcf_rank{1..K}_subformulae/ inside")
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-peaks", type=int, default=100)
    p.add_argument("--test-only", action="store_true", default=True)
    p.add_argument("--start-idx", type=int, default=None)
    p.add_argument("--end-idx", type=int, default=None)


def cmd_subformulae(args: argparse.Namespace) -> None:
    import sys
    from massspecgym.preprocessing.subformulae import main as subformulae_main
    # Rebuild sys.argv so subformulae main() parses correctly
    argv = [
        "subformulae",
        "--predictions", str(args.predictions),
        "--spec-dir", str(args.spec_dir),
        "--labels", str(args.labels),
        "--split", str(args.split),
        "--output-base", str(args.output_base),
        "--max-k", str(args.max_k),
        "--num-workers", str(args.num_workers),
        "--max-peaks", str(args.max_peaks),
    ]
    if args.test_only:
        argv.append("--test-only")
    if args.start_idx is not None:
        argv += ["--start-idx", str(args.start_idx)]
    if args.end_idx is not None:
        argv += ["--end-idx", str(args.end_idx)]
    sys.argv[1:] = argv[1:]
    subformulae_main()


# ============================================================
# build-cache
# ============================================================

def add_build_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-bits", type=int, default=4096)


def cmd_build_cache(args: argparse.Namespace) -> None:
    import numpy as np
    from tqdm import tqdm
    from rdkit.Chem import AllChem, MolFromSmiles
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def smi_to_fp(smi):
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=args.n_bits)
        arr = np.zeros(args.n_bits, dtype=np.uint8)
        for b in bv.GetOnBits():
            arr[b] = 1
        return arr

    log.info(f"Loading candidates from {args.candidates}...")
    with open(args.candidates) as f:
        cands = json.load(f)

    all_smis = set(cands.keys())
    for v in cands.values():
        all_smis.update(v)
    all_smis = sorted(all_smis)
    log.info(f"  {len(all_smis)} unique SMILES")

    fps_list, valid_smis = [], []
    for smi in tqdm(all_smis, desc="Morgan fps"):
        fp = smi_to_fp(smi)
        if fp is not None:
            fps_list.append(fp)
            valid_smis.append(smi)

    fps_arr = np.stack(fps_list)
    fps_packed = np.packbits(fps_arr, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.out), smiles=np.array(valid_smis, dtype=object), fps=fps_packed)
    log.info(f"Saved {len(valid_smis)} fps to {args.out}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="subcommand", required=True)

    add_train_args(subs.add_parser("train", help="Train/test MIST retrieval end-to-end"))
    add_predict_args(subs.add_parser("predict", help="Predict fingerprints from MIST encoder"))
    add_eval_args(subs.add_parser("eval", help="Evaluate precomputed FPs vs candidates"))
    add_formula_args(subs.add_parser("formula", help="Predict formulas with MistCFNet"))
    add_subformulae_args(subs.add_parser("subformulae", help="Generate subformulae JSONs"))
    add_build_cache_args(subs.add_parser("build-cache", help="Pre-build Morgan FP cache (.npz)"))

    args = parser.parse_args()
    dispatch = {
        "train": cmd_train,
        "predict": cmd_predict,
        "eval": cmd_eval,
        "formula": cmd_formula,
        "subformulae": cmd_subformulae,
        "build-cache": cmd_build_cache,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
