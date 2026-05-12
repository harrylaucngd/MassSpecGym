"""
MIST-featurized retrieval dataset for MISTFingerprintRetrieval.
"""
from __future__ import annotations

import json
import typing as T
from pathlib import Path

import matchms
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import massspecgym.utils as utils
from massspecgym.data.datasets import RetrievalDataset
from massspecgym.data.mist_data_mixin import MISTDataMixin

_LABELS_CHUNK_ROWS = 50_000
from massspecgym.data.transforms import MolFingerprinter
from massspecgym.models.encoders.mist.chem_constants import (
    VALID_ELEMENTS, formula_to_dense, ion_remap, ION_LST, get_instr_idx,
)

_N_ELEMENTS = len(VALID_ELEMENTS)
_CLS_TYPE = 3
_FRAG_TYPE = 0
_ION_TO_IDX: dict[str, int] = {ion: i for i, ion in enumerate(ION_LST)}
_MIST_KEYS = {"num_peaks", "types", "instruments", "ion_vec", "form_vec", "intens"}

_DUMMY_SPECTRUM = matchms.Spectrum(
    mz=np.array([0.0], dtype=np.float64),
    intensities=np.array([1.0], dtype=np.float64),
    metadata={"precursor_mz": 0.0},
)


class _NullSpecTransform:
    """Placeholder so MassSpecDataset does not crash when real spec data is absent."""
    def __call__(self, spec):
        return torch.zeros(1)


class MISTRetrievalDataset(MISTDataMixin, RetrievalDataset):
    """RetrievalDataset that featurizes spectra via MIST subformulae JSONs.

    Accepts either the full MassSpecGym TSV or the lighter MIST-format
    labels.tsv (columns: spec, ionization, formula, smiles, inchikey, instrument).
    When a MIST-format labels.tsv is given the spectrum rows are never parsed —
    all spectral features come from the subformulae JSON files.

    The split file (name/split columns) is merged into metadata so that
    MassSpecDataModule can be called with split_pth=None.

    Args:
        subform_folder: Directory of per-spectrum JSON files.
        mist_split_pth: Path to split.tsv with columns name/split or
            identifier/fold. Merged into metadata as 'fold'.
        max_peaks: Max fragment peaks per spectrum (excluding CLS token).
        fp_size: Morgan fingerprint bits for mol/candidate featurization.
        **kwargs: Forwarded to RetrievalDataset (pth, candidates_pth, …).
    """

    def __init__(
        self,
        subform_folder: T.Union[str, Path],
        mist_split_pth: T.Optional[T.Union[str, Path]] = None,
        max_peaks: int = 60,
        fp_size: int = 4096,
        **kwargs,
    ):
        self.subform_folder = Path(subform_folder)
        self.mist_split_pth = Path(mist_split_pth) if mist_split_pth else None
        self.max_peaks = max_peaks
        kwargs["spec_transform"] = _NullSpecTransform()
        kwargs.setdefault("mol_transform", MolFingerprinter(fp_size=fp_size))
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        pth = Path(self.pth) if isinstance(self.pth, str) else self.pth

        if pth is None or (pth.suffix == ".tsv" and self._is_full_msg_tsv(pth)):
            # Full MassSpecGym TSV (or None → HuggingFace download) — delegate
            super().load_data()
            self._merge_split_into_metadata()
            return

        # MIST-format labels.tsv ----------------------------------------
        rename_map = {
            "spec": "identifier",
            "ionization": "adduct",
            "instrument": "instrument_type",
        }
        if self.identifiers_subset is not None:
            allow = set(map(str, self.identifiers_subset))
            chunks: T.List[pd.DataFrame] = []
            for chunk in pd.read_csv(pth, sep="\t", chunksize=_LABELS_CHUNK_ROWS):
                chunk = chunk.rename(columns=rename_map)
                chunk = chunk[chunk["identifier"].astype(str).isin(allow)]
                if len(chunk):
                    chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        else:
            df = pd.read_csv(pth, sep="\t").rename(columns=rename_map)

        if "precursor_mz" not in df.columns:
            df["precursor_mz"] = 0.0

        # Merge fold from split file
        if self.mist_split_pth is not None:
            split = pd.read_csv(self.mist_split_pth, sep="\t")
            split = split.rename(columns={"name": "identifier", "split": "fold"})
            df = df.merge(split[["identifier", "fold"]], on="identifier", how="left")
            df["fold"] = df["fold"].fillna("train")

        self.metadata = df
        self.spectra = pd.Series([_DUMMY_SPECTRUM] * len(df))

        self._load_candidates()

    @staticmethod
    def _is_full_msg_tsv(pth: Path) -> bool:
        """Peek at the header to check for the mzs column."""
        with open(pth) as f:
            header = f.readline()
        return "mzs" in header.split("\t")

    def _merge_split_into_metadata(self):
        """After a full-TSV load, merge fold from MIST split file if needed."""
        if self.mist_split_pth is None or "fold" in self.metadata.columns:
            return
        split = pd.read_csv(self.mist_split_pth, sep="\t")
        split = split.rename(columns={"name": "identifier", "split": "fold"})
        self.metadata = self.metadata.merge(split[["identifier", "fold"]], on="identifier", how="left")
        self.metadata["fold"] = self.metadata["fold"].fillna("train")

    # ------------------------------------------------------------------
    # Item featurization
    # ------------------------------------------------------------------

    def _featurize_one(self, identifier: str, instrument: str) -> dict[str, torch.Tensor]:
        with open(self.subform_folder / f"{identifier}.json") as f:
            d = json.load(f)

        ion = ion_remap.get(d["cand_ion"], ION_LST[0])
        ion_idx = _ION_TO_IDX.get(ion, 0)
        instr_idx = get_instr_idx(instrument)

        output_tbl = d.get("output_tbl")
        if output_tbl is None:
            formulas: T.List[T.Any] = []
            intens_raw: T.List[T.Any] = []
        else:
            formulas = output_tbl.get("formula")
            intens_raw = output_tbl.get("ms2_inten")
            if formulas is None:
                formulas = []
            elif not isinstance(formulas, (list, tuple)):
                formulas = [formulas]
            if intens_raw is None:
                intens_raw = []
            elif not isinstance(intens_raw, (list, tuple)):
                intens_raw = [intens_raw]
        n_pair = min(len(formulas), len(intens_raw))
        formulas = list(formulas[:n_pair])
        intens_raw = list(intens_raw[:n_pair])
        n_frags = min(n_pair, self.max_peaks)

        parent_vec = formula_to_dense(d["cand_form"])
        frag_vecs = np.zeros((n_frags, _N_ELEMENTS), dtype=np.int64)
        frag_intens = np.zeros(n_frags, dtype=np.float32)
        for i in range(n_frags):
            try:
                frag_vecs[i] = formula_to_dense(formulas[i])
            except Exception:
                pass
            frag_intens[i] = float(intens_raw[i])

        n_total = 1 + n_frags
        form_vecs = np.vstack([parent_vec[None], frag_vecs])
        intens = np.concatenate([[1.0], frag_intens])
        types = np.array([_CLS_TYPE] + [_FRAG_TYPE] * n_frags, dtype=np.int64)

        return {
            "num_peaks":   torch.tensor(n_total,   dtype=torch.long),
            "types":       torch.tensor(types,     dtype=torch.long),
            "instruments": torch.tensor(instr_idx, dtype=torch.long),
            "ion_vec":     torch.full((n_total,), ion_idx, dtype=torch.long),
            "form_vec":    torch.tensor(form_vecs, dtype=torch.long),
            "intens":      torch.tensor(intens,    dtype=torch.float32),
        }

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        item.pop("spec", None)

        row = self.metadata.iloc[i]
        identifier = str(row["identifier"])
        instrument = str(row.get("instrument_type", row.get("instrument", "")))
        item.update(self._featurize_one(identifier, instrument))
        return item

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: T.List[dict]) -> dict:
        stripped = [{k: v for k, v in item.items() if k not in _MIST_KEYS} for item in batch]
        collated = RetrievalDataset.collate_fn(stripped)

        collated["num_peaks"]   = torch.stack([item["num_peaks"]   for item in batch])
        collated["instruments"] = torch.stack([item["instruments"] for item in batch])

        max_n = max(item["types"].shape[0] for item in batch)
        collated["types"]    = torch.stack([
            F.pad(item["types"],   (0, max_n - item["types"].shape[0]))   for item in batch])
        collated["ion_vec"]  = torch.stack([
            F.pad(item["ion_vec"], (0, max_n - item["ion_vec"].shape[0])) for item in batch])
        collated["intens"]   = torch.stack([
            F.pad(item["intens"],  (0, max_n - item["intens"].shape[0]))  for item in batch])
        collated["form_vec"] = torch.stack([
            F.pad(item["form_vec"], (0, 0, 0, max_n - item["form_vec"].shape[0])) for item in batch])

        return collated
