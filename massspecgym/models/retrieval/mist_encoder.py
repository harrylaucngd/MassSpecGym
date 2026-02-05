from __future__ import annotations

import typing as T
from pathlib import Path

import torch

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.mist import SpectraEncoderGrowing, load_torch_state_dict
from massspecgym.mist.encoder import threshold_fingerprint


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    if not all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _normalize_encoder_state_dict(sd: dict) -> dict:
    """
    Return a state dict whose keys start with 'spectra_encoder.'.

    Checkpoints may store the encoder under prefixes like 'encoder.' or 'model.encoder.'.
    """
    anchor = "spectra_encoder.0.intermediate_layer.input_layer.weight"

    if anchor in sd:
        return sd

    # Fast path: try common prefixes.
    for pref in ("encoder.", "model.encoder.", "model.module.encoder.", "module.encoder.", "module."):
        key = pref + anchor
        if key in sd:
            return {k[len(pref) :]: v for k, v in sd.items() if k.startswith(pref)}

    # Generic: find a key that ends with anchor and strip that prefix.
    matches = [k for k in sd.keys() if k.endswith(anchor)]
    if not matches:
        raise ValueError(
            "Unable to locate encoder weights in checkpoint: missing anchor key "
            f"ending with '{anchor}'."
        )
    pref = matches[0][: -len(anchor)]
    return {k[len(pref) :]: v for k, v in sd.items() if k.startswith(pref)}


DEFAULT_MSG_ENCODER_KWARGS: dict[str, T.Any] = {
    # Mirrors DiffMS `diffusion_model_spec2mol.py` encoder construction for MSG.
    "inten_transform": "float",
    "peak_attn_layers": 2,
    "num_heads": 8,
    "pairwise_featurization": True,
    "embed_instrument": False,
    "set_pooling": "cls",
    "form_embedder": "pos-cos",
    "output_size": 4096,
    "hidden_size": 512,
    "spectra_dropout": 0.1,
    "top_layers": 1,
    "refine_layers": 4,
    "magma_modulo": 2048,
    # Checkpoints may have been trained with an instrument embedding table larger than our
    # local instrument mapping (we only use indices 0..N-1); keep dim consistent with ckpt.
    "instr_dim": 6,
}


class MistEncoderRetrieval(RetrievalMassSpecGymModel):
    """
    Retrieval inference for the DiffMS/MIST spectra encoder on MSG subformula-assigned inputs.

    Important: The MIST encoder is run in "single-spectrum" mode. Even if the DataLoader batches
    multiple samples, the encoder forward is invoked with batch size 1 per spectrum.
    """

    def __init__(
        self,
        mist_ckpt_path: str | Path,
        *,
        threshold: float = 0.187,
        sanity_shuffle_candidates: bool = False,
        encoder_kwargs: T.Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mist_ckpt_path = str(mist_ckpt_path)
        self.threshold = float(threshold)
        self.sanity_shuffle_candidates = bool(sanity_shuffle_candidates)

        # Load state dict and normalize keys to encoder-only state dict.
        sd = load_torch_state_dict(self.mist_ckpt_path)
        sd = _strip_prefix_if_present(sd, "model.")
        sd = _strip_prefix_if_present(sd, "encoder.")
        sd = _normalize_encoder_state_dict(sd)

        user_kwargs = dict(encoder_kwargs or {})
        merged_kwargs = {**DEFAULT_MSG_ENCODER_KWARGS, **user_kwargs}
        self.encoder_kwargs = merged_kwargs

        self.encoder = SpectraEncoderGrowing(**merged_kwargs)
        self._load_encoder_weights_from_state_dict(sd)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _load_encoder_weights_from_state_dict(self, sd: dict) -> None:
        model_sd = self.encoder.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape}

        if len(filtered) < 0.8 * max(1, len(model_sd)):
            raise ValueError(
                "Failed to load MIST encoder checkpoint (too few matching keys). "
                f"matched={len(filtered)}/{len(model_sd)} from {self.mist_ckpt_path}. "
                "Checkpoint and model architecture likely mismatch."
            )
        self.encoder.load_state_dict(filtered, strict=False)

    @torch.no_grad()
    def _encode_one(self, mist_input: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        # Ensure per-spectrum inference (B=1).
        if "num_peaks" not in mist_input or int(mist_input["num_peaks"].shape[0]) != 1:
            raise ValueError("MIST input must have batch size 1 (num_peaks.shape[0] == 1).")

        inp = {k: v.to(device=device, non_blocking=True) for k, v in mist_input.items()}
        fp_prob, _ = self.encoder(inp)
        if fp_prob.ndim != 2 or fp_prob.shape[0] != 1:
            raise ValueError(f"Unexpected encoder output shape: {tuple(fp_prob.shape)} (expected [1,4096]).")
        fp_bin = threshold_fingerprint(fp_prob, threshold=self.threshold)
        return fp_bin.squeeze(0)

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict[str, torch.Tensor]:
        mist_inputs: list[dict[str, torch.Tensor]] = batch["mist_input"]
        batch_ptr: torch.Tensor = batch["batch_ptr"].to(self.device)
        candidates_fp: torch.Tensor = batch["candidates_mol"].to(device=self.device, dtype=torch.float32)

        fp_pred = torch.stack([self._encode_one(mi, device=self.device) for mi in mist_inputs], dim=0)
        fp_pred = fp_pred.to(dtype=candidates_fp.dtype)

        if self.sanity_shuffle_candidates:
            # Shuffle candidate fingerprints within each sample segment to sanity-check for leakage.
            offsets = torch.cumsum(batch_ptr, dim=0)
            starts = torch.cat([torch.zeros(1, device=offsets.device, dtype=offsets.dtype), offsets[:-1]])
            shuffled = candidates_fp.clone()
            for s, e in zip(starts.tolist(), offsets.tolist()):
                if e - s <= 1:
                    continue
                perm = torch.randperm(e - s, device=shuffled.device)
                shuffled[s:e] = shuffled[s:e][perm]
            candidates_fp = shuffled

        fp_pred_rep = fp_pred.repeat_interleave(batch_ptr, dim=0)
        # Tanimoto similarity on (binary) Morgan fingerprints:
        # tanimoto(a,b) = |a ∩ b| / |a ∪ b| = (a·b) / (sum(a) + sum(b) - a·b)
        # Both fingerprints are expected to be 0/1 vectors in {0,1}.
        intersection = (fp_pred_rep * candidates_fp).sum(dim=-1)
        union = fp_pred_rep.sum(dim=-1) + candidates_fp.sum(dim=-1) - intersection
        scores = intersection / union.clamp_min(1e-8)

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return {"loss": loss, "scores": scores}

    def configure_optimizers(self):
        return None
