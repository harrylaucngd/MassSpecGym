"""
Self-contained MIST (DiffMS) encoder inference utilities for MassSpecGym.

This package must not depend on the external `DiffMS/` folder. It re-implements
only the parts required for loading `checkpoints/encoder_msg.pt` and performing
single-spectrum (no batching) inference for retrieval evaluation.
"""

from .checkpoint import load_torch_state_dict
from .encoder import SpectraEncoderGrowing
from .featurizer import MsgSubformulaFeaturizer, MsgSubformulaSample
from .msg_retrieval_dataset import MsgMistRetrievalDataset

__all__ = [
    "load_torch_state_dict",
    "SpectraEncoderGrowing",
    "MsgSubformulaFeaturizer",
    "MsgSubformulaSample",
    "MsgMistRetrievalDataset",
]
