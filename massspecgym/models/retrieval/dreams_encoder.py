from __future__ import annotations

import typing as T
from pathlib import Path

import torch
import torch.nn.functional as F

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class DreamsFingerprintRetrieval(RetrievalMassSpecGymModel):
    """
    Retrieval model using a fine-tuned DreaMS downstream checkpoint that predicts 4096-dim Morgan fingerprints.

    The model encodes spectra into fingerprint vectors and ranks candidate molecules using cosine similarity
    against candidate Morgan fingerprints provided by `RetrievalDataset` with `MolFingerprinter(fp_size=4096)`.

    Notes:
    - Requires the `dreams` package (DreaMS repo) to be installed in the same environment.
    - For formula-bonus retrieval, use `RetrievalDataset(candidates_pth='bonus', ...)`.
    """

    def __init__(
        self,
        dreams_ckpt_path: str | Path,
        *,
        ssl_backbone_ckpt_path: T.Optional[str | Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dreams_ckpt_path = str(dreams_ckpt_path)
        self.ssl_backbone_ckpt_path = str(ssl_backbone_ckpt_path) if ssl_backbone_ckpt_path else None

        try:
            from dreams.models.heads.heads import FingerprintHead  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import `dreams` (DreaMS). Install it first, e.g.: `pip install -e /path/to/DreaMS-MSG`."
            ) from e

        ckpt = Path(self.dreams_ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"DreaMS checkpoint not found: {ckpt}")

        # Load the fine-tuned downstream module.
        # The checkpoint may have been saved with a `backbone` hyperparameter pointing to the SSL checkpoint.
        # If that path is not valid in this environment, allow overriding it via `ssl_backbone_ckpt_path`.
        load_kwargs = {}
        if self.ssl_backbone_ckpt_path is not None:
            load_kwargs["backbone"] = Path(self.ssl_backbone_ckpt_path)

        try:
            self.encoder = FingerprintHead.load_from_checkpoint(self.dreams_ckpt_path, **load_kwargs)
        except Exception as e:
            hint = ""
            if self.ssl_backbone_ckpt_path is None:
                hint = (
                    " If this checkpoint was saved with a `backbone` hyperparameter pointing to a SSL backbone path "
                    "that is not valid in this environment, pass --dreams_ssl_backbone_ckpt_path to scripts/run.py "
                    "(or `ssl_backbone_ckpt_path=` in code) to override it."
                )
            raise RuntimeError(f"Failed to load DreaMS downstream checkpoint: {self.dreams_ckpt_path}.{hint}") from e
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self._moved_to_device = False

    @torch.no_grad()
    def _maybe_move_to_device(self) -> None:
        if self._moved_to_device:
            return
        self.encoder.to(self.device)
        self._moved_to_device = True

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict[str, torch.Tensor]:
        self._maybe_move_to_device()

        spec = batch["spec"].to(device=self.device, dtype=torch.float32, non_blocking=True)
        batch_ptr: torch.Tensor = batch["batch_ptr"].to(device=self.device)

        # Candidate Morgan fingerprints of shape (sum_candidates, 4096).
        candidates_fp = batch["candidates_mol"].to(device=self.device, dtype=torch.float32, non_blocking=True)

        # Encode spectra -> fingerprint prediction.
        fp_pred = self.encoder(spec, charge=None)
        if fp_pred.ndim != 2:
            raise ValueError(f"Unexpected fingerprint prediction shape: {tuple(fp_pred.shape)}")
        # Convert logits-like outputs to [0,1] fingerprint probabilities (consistent with MassSpecGym baselines).
        fp_pred = torch.sigmoid(fp_pred)

        fp_pred_rep = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = F.cosine_similarity(fp_pred_rep, candidates_fp, dim=-1)

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return {"loss": loss, "scores": scores}

    def configure_optimizers(self):
        return None
