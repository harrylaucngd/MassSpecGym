from __future__ import annotations

import typing as T
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingStoreMetadata:
    model_name: str
    normalized: bool
    dtype: str
    dim: int


class SmilesEmbeddingStore:
    """
    Disk-backed store for SMILES embeddings:
      - smiles: list[str]
      - embeddings: torch.Tensor [N, D]
    """

    def __init__(self, smiles: list[str], embeddings, metadata: EmbeddingStoreMetadata):
        self.smiles = smiles
        self.embeddings = embeddings
        self.metadata = metadata
        self._index = {s: i for i, s in enumerate(smiles)}

    @staticmethod
    def save(path: str | Path, smiles: list[str], embeddings, metadata: EmbeddingStoreMetadata) -> None:
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise ImportError("Saving embeddings requires PyTorch.") from e
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "smiles": smiles,
            "embeddings": embeddings.cpu(),
            "metadata": {
                "model_name": metadata.model_name,
                "normalized": bool(metadata.normalized),
                "dtype": str(metadata.dtype),
                "dim": int(metadata.dim),
            },
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str | Path, map_location: str = "cpu") -> "SmilesEmbeddingStore":
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise ImportError("Loading embeddings requires PyTorch.") from e

        path = Path(path)
        raw = torch.load(path, map_location=map_location)
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected embedding store type: {type(raw)}")
        smiles = raw.get("smiles")
        embeddings = raw.get("embeddings")
        meta = raw.get("metadata") or {}
        if not isinstance(smiles, list) or embeddings is None:
            raise ValueError("Invalid embedding store payload.")
        md = EmbeddingStoreMetadata(
            model_name=str(meta.get("model_name", "")),
            normalized=bool(meta.get("normalized", False)),
            dtype=str(meta.get("dtype", "")),
            dim=int(meta.get("dim", getattr(embeddings, "shape", [0, 0])[1])),
        )
        return SmilesEmbeddingStore(smiles=smiles, embeddings=embeddings, metadata=md)

    def get(self, smiles: list[str]):
        import torch

        idx = []
        for s in smiles:
            if s not in self._index:
                raise KeyError(f"Missing SMILES in embedding store: {s}")
            idx.append(self._index[s])
        idx_t = torch.tensor(idx, device=self.embeddings.device, dtype=torch.long)
        return self.embeddings.index_select(0, idx_t)

