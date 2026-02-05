from __future__ import annotations

from pathlib import Path


def load_torch_state_dict(path: str | Path) -> dict:
    """
    Load a PyTorch state dict from disk safely when possible.
    """
    import torch

    path = Path(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and all(hasattr(v, "shape") for v in obj.values()):
        return obj
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model_state", "encoder_state_dict", "encoder"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
