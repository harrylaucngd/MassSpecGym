from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .chem_utils import VALID_MONO_MASSES, formula_to_dense, get_instr_idx, get_ion_idx


@dataclass(frozen=True)
class MsgSubformulaSample:
    identifier: str
    smiles: str
    instrument: str


class MsgSubformulaFeaturizer:
    """
    Build MIST encoder inputs from a MSG subformula assignment JSON file.

    The JSON is expected to match DiffMS PeakFormula tree format:
      - cand_form: str
      - cand_ion: str
      - output_tbl: {formula: list[str], ms2_inten: list[float], ions: list[str]} or None
    """

    cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}
    cls_type = cat_types["cls"]

    def __init__(self, subform_folder: str | Path, cls_mode: str = "ms1"):
        self.subform_folder = Path(subform_folder)
        self.cls_mode = cls_mode

    def _load_tree(self, identifier: str) -> dict:
        p = self.subform_folder / f"{identifier}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing subformula JSON: {p}")
        with open(p, "r") as f:
            return json.load(f)

    def featurize_one(self, identifier: str, instrument: str) -> dict[str, torch.Tensor]:
        tree = self._load_tree(identifier)

        root_form = tree.get("cand_form")
        root_ion = tree.get("cand_ion")
        out_tbl = tree.get("output_tbl")

        frags = [] if out_tbl is None else (out_tbl.get("formula") or [])
        intens = [] if out_tbl is None else (out_tbl.get("ms2_inten") or [])
        ions = [] if out_tbl is None else (out_tbl.get("ions") or [])

        forms_vec = [formula_to_dense(f) for f in frags]
        root_vec = formula_to_dense(root_form)
        root_ion_idx = get_ion_idx(root_ion)
        root_mass = float((root_vec.astype(np.float64) * VALID_MONO_MASSES).sum())

        inten_vec = [float(x) for x in intens]
        ion_vec = [get_ion_idx(x) for x in ions]
        type_vec = [self.cat_types["frags"]] * len(forms_vec)

        # Append CLS token
        if self.cls_mode == "ms1":
            inten_vec.append(1.0)
            type_vec.append(self.cls_type)
            forms_vec.append(root_vec)
            ion_vec.append(root_ion_idx)
        elif self.cls_mode == "zeros":
            inten_vec.append(0.0)
            type_vec.append(self.cls_type)
            forms_vec.append(np.zeros_like(root_vec))
            ion_vec.append(root_ion_idx)
        else:
            raise ValueError(f"Unknown cls_mode: {self.cls_mode}")

        forms = torch.from_numpy(np.asarray(forms_vec, dtype=np.float32))
        types = torch.tensor(type_vec, dtype=torch.long)
        ions_t = torch.tensor(ion_vec, dtype=torch.long)
        intens_t = torch.tensor(inten_vec, dtype=torch.float32)
        num_peaks = torch.tensor([forms.shape[0]], dtype=torch.long)
        instruments = torch.tensor([get_instr_idx(instrument)], dtype=torch.long)

        # Add batch dimension (B=1) to match encoder expectations.
        return {
            "types": types.unsqueeze(0),
            "form_vec": forms.unsqueeze(0),
            "ion_vec": ions_t.unsqueeze(0),
            "intens": intens_t.unsqueeze(0),
            "num_peaks": num_peaks,
            "instruments": instruments,
        }

