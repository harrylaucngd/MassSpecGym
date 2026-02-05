from __future__ import annotations

import typing as T
from pathlib import Path

import torch
import torch.nn.functional as F

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.specbridge import (
    DreamsAdapter,
    DreamsConfig,
    DreamsEncoder,
    ProcrustesResidualMapper,
    load_specbridge_checkpoint,
)
from massspecgym.specbridge.checkpoint import filter_state_dict_for_module
from massspecgym.specbridge.embedding_store import SmilesEmbeddingStore


class SpecBridgeRetrieval(RetrievalMassSpecGymModel):
    """
    SpecBridge retrieval inference inside MassSpecGym.

    This model expects `RetrievalDataset(..., mol_transform=None)` so candidates are provided as SMILES
    via `batch["candidates_smiles"]` and `batch["batch_ptr"]`.

    It computes a mapped spectrum embedding in the ChemBERTa space and scores candidates using
    cosine similarity (default) or raw dot product.
    """

    def __init__(
        self,
        specbridge_ckpt_path: str | Path,
        candidate_embeddings_path: str | Path,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.specbridge_ckpt_path = str(specbridge_ckpt_path)
        self.candidate_embeddings_path = str(candidate_embeddings_path)
        self.normalize = bool(normalize)

        ckpt = load_specbridge_checkpoint(self.specbridge_ckpt_path)
        args = ckpt.args or {}

        # --- Build spectrum encoder (DreaMS backbone + adapter projection)
        cfg = DreamsConfig()
        dreams = DreamsEncoder(cfg)

        cond_dim = int(args.get("cond_dim", 512))
        mapper_hidden = int(args.get("mapper_hidden", 0))
        n_blocks = int(args.get("n_blocks", args.get("n-blocks", 8)))
        gaussian = not bool(args.get("no_gaussian", False))

        self.spec = DreamsAdapter(
            dreams_encoder=dreams,
            d_out=cond_dim,
            hidden=mapper_hidden,
            freeze_backbone=True,
        )

        # Candidate embeddings store (CPU by default; moved to device on demand)
        self.cand_store = SmilesEmbeddingStore.load(self.candidate_embeddings_path, map_location="cpu")

        # --- Mapper into candidate embedding space
        mol_hidden_size = int(args.get("mol_hidden_size", self.cand_store.metadata.dim))
        self.mapB = ProcrustesResidualMapper(
            d_in=cond_dim,
            d_out=mol_hidden_size,
            n_blocks=n_blocks,
            hidden=mapper_hidden,
            gaussian=gaussian,
        )

        # Load weights from checkpoint (shape-filtered)
        self._load_from_checkpoint(ckpt.model_state)

        for p in self.parameters():
            p.requires_grad = False

    def _load_from_checkpoint(self, model_state: dict) -> None:
        # spec adapter
        spec_keys_total = len(self.spec.proj.state_dict())
        spec_sd = filter_state_dict_for_module(self.spec.proj.state_dict(), model_state, prefix="spec.proj")
        self.spec.proj.load_state_dict(spec_sd, strict=False)
        if spec_keys_total and len(spec_sd) < 0.5 * spec_keys_total:
            raise ValueError(
                "Failed to load SpecBridge spectrum projection weights: "
                f"loaded={len(spec_sd)}/{spec_keys_total}. "
                "Checkpoint and model architecture likely mismatch."
            )

        # dreams backbone
        dreams_keys_total = len(self.spec.dreams.state_dict())
        dreams_sd = filter_state_dict_for_module(self.spec.dreams.state_dict(), model_state, prefix="spec.dreams")
        self.spec.dreams.load_state_dict(dreams_sd, strict=False)
        if dreams_keys_total and len(dreams_sd) < 0.5 * dreams_keys_total:
            raise ValueError(
                "Failed to load SpecBridge DreaMS backbone weights: "
                f"loaded={len(dreams_sd)}/{dreams_keys_total}. "
                "Checkpoint and model architecture likely mismatch."
            )

        # mapper
        map_w_sd = filter_state_dict_for_module(self.mapB.W.state_dict(), model_state, prefix="mapB.W")
        self.mapB.W.load_state_dict(map_w_sd, strict=False)
        # blocks
        blocks_keys_total = len(self.mapB.blocks.state_dict())
        blocks_sd = filter_state_dict_for_module(self.mapB.blocks.state_dict(), model_state, prefix="mapB.blocks")
        self.mapB.blocks.load_state_dict(blocks_sd, strict=False)
        if blocks_keys_total and len(blocks_sd) < 0.5 * blocks_keys_total:
            raise ValueError(
                "Failed to load SpecBridge mapper block weights: "
                f"loaded={len(blocks_sd)}/{blocks_keys_total}. "
                "Checkpoint and model architecture likely mismatch."
            )
        # lv (optional)
        if self.mapB.lv is not None:
            lv_sd = filter_state_dict_for_module(self.mapB.lv.state_dict(), model_state, prefix="mapB.lv")
            self.mapB.lv.load_state_dict(lv_sd, strict=False)

    def _scores_from_embeddings(
        self,
        query: torch.Tensor,  # [B, D]
        candidates: torch.Tensor,  # [sum(C), D]
        batch_ptr: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        if self.normalize:
            query = F.normalize(query, dim=-1)
            candidates = F.normalize(candidates, dim=-1)
        query_rep = query.repeat_interleave(batch_ptr, dim=0)
        return (query_rep * candidates).sum(dim=-1)

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict[str, torch.Tensor]:
        x = batch["spec"]  # [B, N, 2]
        batch_ptr = batch["batch_ptr"]

        # Embed spectrum -> mapped molecule space
        z_s = self.spec(x, meta={"peaks": x})
        mu, _ = self.mapB(z_s)

        # Candidate embeddings lookup
        cand_smiles: list[str] = batch.get("candidates_smiles") or batch.get("candidates_mol")
        cand_emb = self.cand_store.get(cand_smiles).to(device=mu.device, dtype=mu.dtype)

        scores = self._scores_from_embeddings(mu, cand_emb, batch_ptr)

        loss = torch.tensor(0.0, device=mu.device, requires_grad=True)
        return {"loss": loss, "scores": scores}

    def configure_optimizers(self):
        return None
