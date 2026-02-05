from __future__ import annotations

import json
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate

import massspecgym.utils as utils
from massspecgym.data.transforms import MolFingerprinter, MolToInChIKey, MolTransform

from .featurizer import MsgSubformulaFeaturizer


class MsgMistRetrievalDataset(Dataset):
    """
    Retrieval dataset backed by MSG subformula assignments (DiffMS/MIST format).

    This dataset is designed to plug into MassSpecGym's official retrieval evaluation pipeline
    (candidates JSON + HitRate@K + optional MCES@1), while using subformula JSON trees as the
    spectrum input for the MIST encoder.
    """

    def __init__(
        self,
        labels_pth: str | Path,
        split_pth: str | Path,
        subform_folder: str | Path,
        *,
        candidates_pth: str | Path | None = None,
        mol_label_transform: MolTransform = MolToInChIKey(),
        fingerprinter: MolFingerprinter | None = None,
        cls_mode: str = "ms1",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.labels_pth = Path(labels_pth)
        self.split_pth = Path(split_pth)
        self.subform_folder = Path(subform_folder)
        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform
        self.fingerprinter = fingerprinter or MolFingerprinter(fp_size=4096, radius=2)
        self.cls_mode = cls_mode
        self.dtype = dtype

        self.featurizer = MsgSubformulaFeaturizer(self.subform_folder, cls_mode=self.cls_mode)

        self._fp_cache: dict[str, np.ndarray] = {}
        self._label_cache: dict[str, T.Any] = {}

        self._load_metadata()
        self._load_candidates()

    def _load_metadata(self) -> None:
        if not self.labels_pth.is_file():
            raise FileNotFoundError(f"Missing labels file: {self.labels_pth}")
        if not self.split_pth.is_file():
            raise FileNotFoundError(f"Missing split file: {self.split_pth}")

        labels = pd.read_csv(self.labels_pth, sep="\t")
        split = pd.read_csv(self.split_pth, sep="\t")

        labels_cols = set(labels.columns)
        split_cols = set(split.columns)

        # ---- Normalize labels columns (DiffMS uses `spec`; MassSpecGym uses `identifier`)
        if "spec" in labels_cols:
            labels = labels.rename(columns={"spec": "identifier"})
        elif "identifier" not in labels_cols:
            raise ValueError(
                "`labels.tsv` must contain either a 'spec' column (DiffMS MSG) or an 'identifier' column. "
                f"Got: {list(labels.columns)}"
            )

        # ---- Normalize split columns (DiffMS uses name/split; MassSpecGym uses identifier/fold)
        if {"name", "split"} <= split_cols:
            split = split.rename(columns={"name": "identifier", "split": "fold"})
        elif {"spec", "fold"} <= split_cols:
            split = split.rename(columns={"spec": "identifier"})
        elif {"identifier", "fold"} <= split_cols:
            pass
        else:
            raise ValueError(
                "`split.tsv` must contain either columns ['name','split'] (DiffMS MSG) or "
                "['identifier','fold'] / ['spec','fold']. "
                f"Got: {list(split.columns)}"
            )

        labels = labels.astype(str)
        split = split.astype(str)

        # Merge and normalize column names to MassSpecGym expectations.
        df = labels.merge(split[["identifier", "fold"]], on="identifier", how="inner")
        if "smiles" not in df.columns:
            raise ValueError("`labels.tsv` must contain a 'smiles' column for retrieval evaluation.")
        if "instrument" not in df.columns:
            df["instrument"] = ""

        self.metadata = pd.DataFrame(
            {
                "identifier": df["identifier"].astype(str),
                "fold": df["fold"].astype(str),
                "smiles": df["smiles"].astype(str),
                "instrument": df["instrument"].astype(str),
            }
        )

    def _resolve_candidates_pth(self, candidates_pth: str | Path | None) -> Path:
        if candidates_pth is None:
            return Path(
                utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_mass.json")
            )
        if candidates_pth == "bonus":
            return Path(
                utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json")
            )
        p = Path(candidates_pth)
        if p.is_file():
            return p
        # Treat as HF Hub filename.
        return Path(utils.hugging_face_download(str(candidates_pth)))

    def _load_candidates(self) -> None:
        candidates_pth = self._resolve_candidates_pth(self.candidates_pth)
        with open(candidates_pth, "r") as f:
            self.candidates: dict[str, list[str]] = json.load(f)

    def __len__(self) -> int:
        return len(self.metadata)

    def _fingerprint(self, smiles: str) -> np.ndarray:
        if smiles in self._fp_cache:
            return self._fp_cache[smiles]
        fp = self.fingerprinter(smiles)
        fp = np.asarray(fp, dtype=np.float32)
        self._fp_cache[smiles] = fp
        return fp

    def _label(self, smiles: str):
        if smiles in self._label_cache:
            return self._label_cache[smiles]
        lab = self.mol_label_transform(smiles)
        self._label_cache[smiles] = lab
        return lab

    def __getitem__(self, i: int) -> dict:
        row = self.metadata.iloc[i]
        identifier = str(row["identifier"])
        smiles = str(row["smiles"])
        instrument = str(row.get("instrument", ""))

        # MIST encoder input (B=1 tensors).
        mist_input = self.featurizer.featurize_one(identifier=identifier, instrument=instrument)

        # Official retrieval candidates.
        if smiles not in self.candidates:
            raise ValueError(
                f"No retrieval candidates for query SMILES='{smiles}'. "
                "This likely indicates a mismatch between your MSG labels SMILES formatting and "
                "the MassSpecGym candidates JSON."
            )
        candidates_smiles = self.candidates[smiles]

        # Labels are defined by matching a canonicalized molecule representation.
        query_label = self._label(smiles)
        labels = [self._label(c) == query_label for c in candidates_smiles]
        if not any(labels):
            raise ValueError("Query molecule not found in its candidate list (no positive label).")

        candidates_fp = torch.as_tensor(
            np.stack([self._fingerprint(c) for c in candidates_smiles], axis=0),
            dtype=self.dtype,
        )

        # Dummy `spec` tensor is provided only to satisfy MassSpecGym's retrieval base class logging.
        spec = torch.zeros((1, 1), dtype=torch.float32)

        return {
            "spec": spec,
            "identifier": identifier,
            "smiles": smiles,
            "candidates_smiles": candidates_smiles,
            "candidates_mol": candidates_fp,
            "labels": labels,
            "mist_input": mist_input,
        }

    @staticmethod
    def _collate_fn_variable_size(batch: T.Sequence[dict], key: str) -> T.Union[torch.Tensor, list]:
        if isinstance(batch[0][key], list):
            collated_item = sum([item[key] for item in batch], start=[])
            if len(collated_item) == 0:
                return collated_item
            if not isinstance(collated_item[0], str):
                collated_item = torch.as_tensor(collated_item)
            return collated_item

        if isinstance(batch[0][key], torch.Tensor):
            return torch.cat([item[key] for item in batch], dim=0)

        raise ValueError(f"Unsupported type for key '{key}': {type(batch[0][key])}")

    @staticmethod
    def collate_fn(batch: T.Sequence[dict]) -> dict:
        collated: dict[str, T.Any] = {}
        for k in batch[0].keys():
            if k == "mist_input":
                collated[k] = [item[k] for item in batch]
            elif k.startswith("candidates") or k == "labels":
                collated[k] = MsgMistRetrievalDataset._collate_fn_variable_size(batch, k)
            else:
                collated[k] = default_collate([item[k] for item in batch])

        collated["batch_ptr"] = torch.as_tensor([len(item["candidates_smiles"]) for item in batch], dtype=torch.long)
        return collated
