from __future__ import annotations

import copy

import torch
import torch.nn as nn

from . import form_embedders, transformer_layer

EPS = 1e-9


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def get_num_inten_feats(inten_transform: str) -> int:
    if inten_transform in {"float", "zero", "log"}:
        return 1
    if inten_transform == "cat":
        return 10
    raise NotImplementedError(inten_transform)


class MLPBlocks(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, num_layers: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(middle_layer, num_layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.dropout_layer(out)
        out = self.activation(out)
        for layer in self.layers:
            out = layer(out)
            out = self.dropout_layer(out)
            out = self.activation(out)
        return out


class FormulaTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        peak_attn_layers: int,
        set_pooling: str = "intensity",
        spectra_dropout: float = 0.1,
        pairwise_featurization: bool = False,
        num_heads: int = 8,
        output_size: int = 2048,
        form_embedder: str = "float",
        embed_instrument: bool = False,
        inten_transform: str = "float",
        no_diffs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_heads = num_heads
        self.dim_feedforward = self.hidden_size * 4
        self.spectra_dropout = spectra_dropout
        self.set_pooling = set_pooling
        self.output_size = output_size
        self.no_diffs = no_diffs

        self.form_embedder = form_embedder
        self.form_embedder_mod = form_embedders.get_embedder(self.form_embedder)

        # Instrument embedding is represented as an identity matrix parameter in DiffMS.
        self.embed_instrument = embed_instrument
        self.instr_dim = int(kwargs.get("instr_dim", 5))  # will be overwritten by checkpoint load
        self.instrument_embedder = nn.Parameter(torch.eye(self.instr_dim))

        self.inten_transform = inten_transform
        self.inten_feats = get_num_inten_feats(self.inten_transform)
        self.num_types = 4
        self.cls_type = 3
        self.adduct_dim = 8

        self.pairwise_featurization = pairwise_featurization

        self.formula_dim = self.form_embedder_mod.full_dim
        self.input_dim = self.formula_dim * 2 + self.num_types + self.instr_dim + self.inten_feats + self.adduct_dim

        self.intermediate_layer = MLPBlocks(
            input_size=self.input_dim, hidden_size=self.hidden_size, dropout=self.spectra_dropout, num_layers=2
        )
        self.pairwise_featurizer = None
        if self.pairwise_featurization:
            self.pairwise_featurizer = MLPBlocks(
                input_size=self.formula_dim, hidden_size=self.hidden_size, dropout=self.spectra_dropout, num_layers=2
            )

        peak_attn_layer = transformer_layer.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attn_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.spectra_dropout,
            pairwise_featurization=pairwise_featurization,
        )
        self.peak_attn_layers = _get_clones(peak_attn_layer, peak_attn_layers)
        self.bin_encoder = None

    def forward(self, batch: dict, return_aux: bool = False):
        num_peaks = batch["num_peaks"]
        peak_types = batch["types"]
        instruments = batch["instruments"]
        adducts = batch["ion_vec"]

        device = num_peaks.device
        batch_dim = num_peaks.shape[0]
        peak_dim = peak_types.shape[-1]

        cls_token_mask = peak_types == self.cls_type

        orig_form_vec = batch["form_vec"][:, :, :]
        form_diffs = orig_form_vec[:, :, None, :] - orig_form_vec[:, None, :, :]

        abs_diffs = form_diffs[cls_token_mask]
        form_vec = self.form_embedder_mod(orig_form_vec)
        diff_vec = self.form_embedder_mod(abs_diffs)
        if self.no_diffs:
            diff_vec = diff_vec.fill_(0)

        intens_temp = batch["intens"]
        if self.inten_transform == "cat":
            raise NotImplementedError("inten_transform=cat is not used in MSG config")
        inten_tensor = intens_temp[:, :, None]

        one_hot_types = nn.functional.one_hot(peak_types, self.num_types)
        one_hot_adducts = nn.functional.one_hot(adducts.long(), self.adduct_dim)

        embedded_instruments = self.instrument_embedder[instruments.long()]
        if self.embed_instrument:
            embedded_instruments = embedded_instruments[:, None, :].repeat(1, peak_dim, 1)
        else:
            embedded_instruments = torch.zeros(batch_dim, peak_dim, self.instr_dim, device=device)

        input_vec = torch.cat(
            [form_vec, diff_vec, one_hot_types, one_hot_adducts, inten_tensor, embedded_instruments], dim=-1
        )
        peak_tensor = self.intermediate_layer(input_vec)

        peak_tensor = peak_tensor.transpose(0, 1)
        peak_dim = peak_tensor.shape[0]
        peaks_arranged = torch.arange(peak_dim, device=device)
        attn_mask = ~(peaks_arranged[None, :] < num_peaks[:, None])

        pairwise_features = None
        if self.pairwise_featurization:
            # same_sign mask used in DiffMS
            same_sign = torch.all(form_diffs >= 0, -1) | torch.all(form_diffs <= 0, -1)
            form_diffs = form_diffs.clone()
            form_diffs[~same_sign].fill_(0)
            form_diffs = torch.abs(form_diffs)
            pairwise_features = self.pairwise_featurizer(self.form_embedder_mod(form_diffs))

        aux_output = {}
        for layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = layer(
                peak_tensor, pairwise_features=pairwise_features, src_key_padding_mask=attn_mask
            )

        output, peak_tensor_out = self._pool_out(peak_tensor, inten_tensor, peak_types, attn_mask, batch_dim)
        aux_output["peak_tensor"] = peak_tensor_out.transpose(0, 1)
        if return_aux:
            return output, aux_output
        return output

    def _pool_out(self, peak_tensor, inten_tensor, peak_types, attn_mask, batch_dim):
        zero_mask = attn_mask[:, :, None].repeat(1, 1, self.hidden_size).transpose(0, 1)
        peak_tensor = peak_tensor.clone()
        peak_tensor[zero_mask] = 0

        if self.set_pooling == "cls":
            pool_factor = (peak_types == self.cls_type).float()
        elif self.set_pooling == "intensity":
            inten_tensor = inten_tensor.reshape(batch_dim, -1)
            intensities_sum = inten_tensor.sum(1).reshape(-1, 1) + EPS
            inten_tensor = inten_tensor / intensities_sum
            pool_factor = inten_tensor * ~attn_mask
        elif self.set_pooling == "mean":
            inten_tensor = inten_tensor.reshape(batch_dim, -1)
            pool_factor = torch.clone(inten_tensor).fill_(1)
            pool_factor = pool_factor * ~attn_mask
            pool_factor[pool_factor == 0] = 1
            pool_factor = pool_factor / pool_factor.sum(1).reshape(-1, 1)
        elif self.set_pooling == "root":
            inten_tensor = inten_tensor.reshape(batch_dim, -1)
            pool_factor = torch.zeros_like(inten_tensor)
            pool_factor[:, 0] = 1
        else:
            raise NotImplementedError(self.set_pooling)

        output = torch.einsum("nbd,bn->bd", peak_tensor, pool_factor)
        return output, peak_tensor


class FPGrowingModule(nn.Module):
    def __init__(self, hidden_input_dim: int = 256, final_target_dim: int = 4096, num_splits: int = 4, reduce_factor: int = 2):
        super().__init__()
        import numpy as np

        self.hidden_input_dim = hidden_input_dim
        self.final_target_dim = final_target_dim
        self.num_splits = num_splits
        self.reduce_factor = reduce_factor

        layer_dims = [int(np.ceil(final_target_dim / (reduce_factor**num_split))) for num_split in range(num_splits + 1)][::-1]
        self.output_dims = layer_dims

        self.initial_predict = nn.Sequential(nn.Linear(hidden_input_dim, layer_dims[0]), nn.Sigmoid())

        predict_bricks = []
        gate_bricks = []
        for layer_dim_ind, layer_dim in enumerate(layer_dims[:-1]):
            out_dim = layer_dims[layer_dim_ind + 1]
            predict_bricks.append(nn.Sequential(nn.Linear(layer_dim, out_dim), nn.Sigmoid()))
            gate_bricks.append(nn.Sequential(nn.Linear(hidden_input_dim, out_dim), nn.Sigmoid()))

        self.predict_bricks = nn.ModuleList(predict_bricks)
        self.gate_bricks = nn.ModuleList(gate_bricks)

    def forward(self, hidden: torch.Tensor):
        cur_pred = self.initial_predict(hidden)
        output_preds = [cur_pred]
        for predict_brick, gate_brick in zip(self.predict_bricks, self.gate_bricks):
            gate_outs = gate_brick(hidden)
            pred_out = predict_brick(cur_pred)
            cur_pred = gate_outs * pred_out
            output_preds.append(cur_pred)
        return output_preds

