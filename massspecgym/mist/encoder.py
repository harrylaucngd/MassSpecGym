from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from . import modules


class SpectraEncoderGrowing(nn.Module):
    """
    MIST spectra encoder (growing fingerprint head) as used by DiffMS for MSG.

    This model outputs a 4096-dim fingerprint probability vector in (0,1).
    """

    def __init__(
        self,
        form_embedder: str = "pos-cos",
        output_size: int = 4096,
        hidden_size: int = 256,
        spectra_dropout: float = 0.1,
        peak_attn_layers: int = 2,
        num_heads: int = 8,
        pairwise_featurization: bool = True,
        set_pooling: str = "cls",
        embed_instrument: bool = False,
        inten_transform: str = "float",
        top_layers: int = 1,
        refine_layers: int = 4,
        magma_modulo: int = 512,
        **kwargs,
    ):
        super().__init__()

        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            num_heads=num_heads,
            pairwise_featurization=pairwise_featurization,
            set_pooling=set_pooling,
            embed_instrument=embed_instrument,
            inten_transform=inten_transform,
            spectra_dropout=spectra_dropout,
            form_embedder=form_embedder,
            **kwargs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))
        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))
        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        spectra_predictor = modules.FPGrowingModule(
            hidden_input_dim=hidden_size,
            final_target_dim=output_size,
            num_splits=refine_layers,
            reduce_factor=2,
        )

        self.spectra_encoder = nn.ModuleList([spectra_encoder_main, fragment_predictor, spectra_predictor])

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)
        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}

        output_list = self.spectra_encoder[2](encoder_output)
        intermediates = output_list[:-1]
        final_output = output_list[-1]

        aux_outputs["int_preds"] = intermediates
        aux_outputs["h0"] = encoder_output
        return final_output, aux_outputs


@torch.no_grad()
def threshold_fingerprint(fp_probs: torch.Tensor, threshold: float = 0.187) -> torch.Tensor:
    """
    Apply the MIST thresholding heuristic used by DiffMS (MIST binarization).
    """
    return (fp_probs >= threshold).to(fp_probs.dtype)
