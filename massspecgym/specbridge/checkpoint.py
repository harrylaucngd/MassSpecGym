from __future__ import annotations

import typing as T
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpecBridgeCheckpoint:
    """
    Lightweight view of a SpecBridge checkpoint saved by `SpecBridge/specbridge/train/train.py`.

    The file is a `torch.save` of a dict with at least:
      - "model": state_dict (flat dict[str, Tensor])
      - "args":  training arguments (dict[str, Any])
    """

    path: Path
    model_state: dict
    args: dict


def _torch_load(path: Path) -> dict:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SpecBridge checkpoint loading requires PyTorch. "
            "Install torch and retry."
        ) from e

    # Prefer weights_only when available to avoid unpickling arbitrary objects.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_specbridge_checkpoint(path: str | Path) -> SpecBridgeCheckpoint:
    path = Path(path)
    raw = _torch_load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(raw)}")

    if "model" not in raw or not isinstance(raw["model"], dict):
        raise ValueError('Checkpoint must contain a dict under key "model".')

    args = raw.get("args") or {}
    if not isinstance(args, dict):
        args = dict(args)

    return SpecBridgeCheckpoint(path=path, model_state=raw["model"], args=args)


def filter_state_dict_for_module(
    module_state_dict: dict[str, T.Any],
    checkpoint_state_dict: dict[str, T.Any],
    prefix: str = "",
) -> dict[str, T.Any]:
    """
    Filter checkpoint parameters to those that exist in the module state dict and match shapes.

    Args:
        module_state_dict: target module.state_dict()
        checkpoint_state_dict: flat checkpoint dict (full model state dict)
        prefix: if provided, only consider ckpt keys starting with f"{prefix}."
    """
    out: dict[str, T.Any] = {}
    for k, v in checkpoint_state_dict.items():
        if prefix:
            if not k.startswith(prefix + "."):
                continue
            kk = k[len(prefix) + 1 :]
        else:
            kk = k
        if kk not in module_state_dict:
            continue
        try:
            if hasattr(v, "shape") and hasattr(module_state_dict[kk], "shape"):
                if tuple(v.shape) != tuple(module_state_dict[kk].shape):
                    continue
        except Exception:
            continue
        out[kk] = v
    return out

