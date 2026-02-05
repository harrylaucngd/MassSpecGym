from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import Atom

# NOTE: This file mirrors the constants used by DiffMS/MIST training so that
# the checkpoint `encoder_msg.pt` can be loaded without shape mismatches.

P_TBL = Chem.GetPeriodicTable()

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = r"([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS: list[str] = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]

VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]

CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = np.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM), dtype=np.float32)
ATOM_NUM_TO_ONEHOT[np.array(VALID_ATOM_NUM), np.arange(CHEM_ELEMENT_NUM)] = 1.0

VALID_MONO_MASSES = np.array([P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS], dtype=np.float64)

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS), dtype=np.float32)
element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, VALID_MONO_MASSES.tolist()))

# Reasonable normalization vector for elements (training-time constant).
NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6], dtype=np.float32)


ION_LST: list[str] = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

ION_REMAP: dict[str, str] = dict(zip(ION_LST, ION_LST))
ION_REMAP.update(
    {
        "[M+NH4]+": "[M+H3N+H]+",
        "M+H": "[M+H]+",
        "M+Na": "[M+Na]+",
        "M+H-H2O": "[M-H2O+H]+",
        "M-H2O+H": "[M-H2O+H]+",
        "M+NH4": "[M+H3N+H]+",
        "M-2H2O+H": "[M-H4O2+H]+",
        "[M-2H2O+H]+": "[M-H4O2+H]+",
    }
)

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST), dtype=np.int64)))


instrument_to_type = defaultdict(lambda: "unknown")
instrument_to_type.update(
    {
        "Thermo Finnigan Velos Orbitrap": "orbitrap",
        "Thermo Finnigan Elite Orbitrap": "orbitrap",
        "Orbitrap Fusion Lumos": "orbitrap",
        "Q-ToF (LCMS)": "qtof",
        "Unknown (LCMS)": "unknown",
        "ion trap": "iontrap",
        "FTICR (LCMS)": "fticr",
        "Bruker Q-ToF (LCMS)": "qtof",
        "Orbitrap (LCMS)": "orbitrap",
    }
)
instruments = sorted(list(set(instrument_to_type.values())))
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments), dtype=np.int64)))


def get_ion_idx(ion: str) -> int:
    ion = (ion or "").strip()
    ion = ION_REMAP.get(ion, ion)
    if ion not in ion_to_idx:
        raise KeyError(f"Unknown ion/adduct: {ion}")
    return int(ion_to_idx[ion])


def get_instr_idx(instrument: str) -> int:
    inst = instrument_to_type.get((instrument or "").strip(), "unknown")
    return int(instrument_to_idx[inst])


def formula_to_dense(chem_formula: str) -> np.ndarray:
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula or ""):
        num_i = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        total_onehot.append(np.repeat(one_hot, repeats=num_i, axis=0))

    if len(total_onehot) == 0:
        return np.zeros(len(element_to_position), dtype=np.float32)
    return np.vstack(total_onehot).sum(0).astype(np.float32, copy=False)

