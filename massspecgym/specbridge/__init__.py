"""
SpecBridge-compatible inference components implemented inside MassSpecGym.

This package is intentionally self-contained and must not depend on the external
`SpecBridge/` folder in this repository. The goal is to allow evaluating
SpecBridge checkpoints using MassSpecGym's official retrieval evaluation.
"""

from .checkpoint import load_specbridge_checkpoint
from .chemberta import ChemBertaEmbedder
from .dreams import DreamsConfig, DreamsEncoder
from .spec_adapter import DreamsAdapter
from .mapper import ProcrustesResidualMapper

__all__ = [
    "load_specbridge_checkpoint",
    "ChemBertaEmbedder",
    "DreamsConfig",
    "DreamsEncoder",
    "DreamsAdapter",
    "ProcrustesResidualMapper",
]

