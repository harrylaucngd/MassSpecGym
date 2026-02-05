from __future__ import annotations

import typing as T
from dataclasses import dataclass


@dataclass(frozen=True)
class ChemBertaConfig:
    model_name: str
    max_length: int = 256


class ChemBertaEmbedder:
    """
    ChemBERTa (or compatible HF model) embedder returning the first-token embedding.

    SpecBridge uses `last_hidden_state[:, 0]` (CLS for RoBERTa/BERT-like models).
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        device: str = "cpu",
        dtype: str | None = None,
    ):
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise ImportError("ChemBertaEmbedder requires PyTorch.") from e
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ChemBertaEmbedder requires `transformers`. "
                "Install `transformers` and retry."
            ) from e

        self.torch = torch
        self.device = torch.device(device)

        torch_dtype = None
        if dtype is not None:
            torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = max_length

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))

    def load_weights_from_state_dict(self, state_dict: dict, prefix: str = "chem_mdl") -> None:
        """
        Load weights from a SpecBridge checkpoint state dict that contains keys like "chem_mdl.*".
        """
        own = self.model.state_dict()
        filtered: dict[str, T.Any] = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix + "."):
                continue
            kk = k[len(prefix) + 1 :]
            if kk in own and hasattr(v, "shape") and tuple(v.shape) == tuple(own[kk].shape):
                filtered[kk] = v
        self.model.load_state_dict(filtered, strict=False)

    def embed(self, smiles: list[str], batch_size: int = 64) -> "torch.Tensor":
        torch = self.torch
        if len(smiles) == 0:
            return torch.empty((0, self.hidden_size), device=self.device)

        outs = []
        for i in range(0, len(smiles), batch_size):
            chunk = smiles[i : i + batch_size]
            toks = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                h = self.model(**toks).last_hidden_state[:, 0]
            outs.append(h)
        return torch.cat(outs, dim=0)

