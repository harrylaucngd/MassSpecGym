"""
MIST fingerprint retrieval: predict Morgan FP from spectrum, rank by Tanimoto.

Uses the MIST encoder (SpectraEncoderGrowing) to predict a 2048-bit molecular
fingerprint from the MS/MS spectrum, then ranks retrieval candidates by
Tanimoto similarity between predicted and candidate fingerprints.

This is a bonus-task retrieval strategy.
"""

from pathlib import Path
import typing as T

import torch
import torch.nn as nn

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.utils import CosSimLoss


class MISTFingerprintRetrieval(RetrievalMassSpecGymModel):
    """MIST-based retrieval via predicted fingerprint similarity.

    Loads a pretrained MIST SpectraEncoderGrowing checkpoint, predicts
    a fingerprint for each query spectrum, and ranks candidates by
    Tanimoto (or cosine) similarity.

    Args:
        encoder_checkpoint: Path to pretrained MIST encoder checkpoint.
        fp_bits: Fingerprint dimensionality (4096 for Morgan).
        similarity: Similarity function ('cosine' or 'tanimoto').
        encoder_kwargs: Architecture overrides for SpectraEncoderGrowing.
            Defaults match encoder_msg.pt.
    """

    def __init__(
        self,
        encoder_checkpoint: T.Optional[str] = None,
<<<<<<< HEAD
        fp_bits: int = 4096,
        similarity: str = "cosine",
        encoder_kwargs: T.Optional[dict] = None,
        fp_save_path: T.Optional[str] = None,
=======
        fp_bits: int = 2048,
        similarity: str = "tanimoto",
>>>>>>> dcdc3fd7f4760d2097ea00a4f272defead043139
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fp_bits = fp_bits
        self.similarity = similarity
        self.fp_save_path = Path(fp_save_path) if fp_save_path else None
        self._fp_buffer: dict = {}
        self.loss_fn = CosSimLoss()

        from massspecgym.models.encoders.mist.encoder import SpectraEncoderGrowing
<<<<<<< HEAD
        enc_kw = dict(
            form_embedder="pos-cos",
            spectra_dropout=0.1,
            inten_transform="float",
            embed_instrument=False,
            set_pooling="cls",
            peak_attn_layers=2,
            num_heads=8,
            refine_layers=4,
            pairwise_featurization=True,
            instr_dim=6,
            output_size=fp_bits,
=======
        self.encoder = SpectraEncoderGrowing(
            form_embedder="pos-cos", output_size=fp_bits, hidden_size=256,
            peak_attn_layers=4, num_heads=8, refine_layers=4,
            set_pooling="cls", pairwise_featurization=True,
>>>>>>> dcdc3fd7f4760d2097ea00a4f272defead043139
        )
        if encoder_checkpoint:
            ckpt = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            enc_kw["hidden_size"] = state_dict["spectra_encoder.0.intermediate_layer.input_layer.bias"].shape[0]
            enc_kw["magma_modulo"] = state_dict["spectra_encoder.1.0.weight"].shape[0]
        if encoder_kwargs:
            enc_kw.update(encoder_kwargs)
        self.encoder = SpectraEncoderGrowing(**enc_kw)
        if encoder_checkpoint:
            self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, batch: dict) -> torch.Tensor:
        fp_pred, _ = self.encoder(batch)
        return fp_pred

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        fp_pred = self.forward(batch)
        fp_true = batch.get("mol")
        if fp_true is not None and fp_true.shape[-1] == fp_pred.shape[-1]:
            loss = self.loss_fn(fp_true, fp_pred)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=fp_pred.device)

        cands = batch.get("candidates_mol", batch.get("candidates"))
        batch_ptr = batch["batch_ptr"]
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)

        if self.similarity == "tanimoto":
            intersection = (fp_pred_repeated * cands).sum(dim=-1)
            union = fp_pred_repeated.sum(dim=-1) + cands.sum(dim=-1) - intersection
            scores = intersection / union.clamp(min=1e-8)
        else:
            scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        if stage == Stage.TEST and self.fp_save_path is not None:
            for ident, fp in zip(batch["identifier"], fp_pred):
                self._fp_buffer[str(ident)] = fp.detach().cpu()

        return dict(loss=loss, scores=scores, processable_mask=batch.get("processable_mask", None))

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.fp_save_path is not None and self._fp_buffer:
            self.fp_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._fp_buffer, self.fp_save_path)
            print(f"Saved {len(self._fp_buffer)} raw fingerprints to {self.fp_save_path}")
            self._fp_buffer = {}

    def on_batch_end(self, outputs, batch, batch_idx, stage):
        # base class uses batch['spec'].size(0) for the batch_size log arg;
        # num_peaks has the same shape (batch_size,) so works as a drop-in
        super().on_batch_end(outputs, {**batch, "spec": batch["num_peaks"]}, batch_idx, stage)
