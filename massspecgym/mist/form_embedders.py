from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .chem_utils import NORM_VEC


class IntFeaturizer(nn.Module):
    MAX_COUNT_INT = 255
    NUM_EXTRA_EMBEDDINGS = 1

    def __init__(self, embedding_dim: int):
        super().__init__()
        weights = torch.zeros(self.NUM_EXTRA_EMBEDDINGS, embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
        self.embedding_dim = embedding_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_shape = tensor.shape
        out_tensor = torch.empty((*orig_shape, self.embedding_dim), device=tensor.device)
        extra_embed = tensor >= self.MAX_COUNT_INT

        tensor = tensor.long()
        norm_embeds = self.int_to_feat_matrix[tensor[~extra_embed]]
        extra_embeds = self._extra_embeddings[tensor[extra_embed] - self.MAX_COUNT_INT]

        out_tensor[~extra_embed] = norm_embeds
        out_tensor[extra_embed] = extra_embeds
        return out_tensor.reshape(*orig_shape[:-1], -1)

    @property
    def num_dim(self) -> int:
        return int(self.int_to_feat_matrix.shape[1])

    @property
    def full_dim(self) -> int:
        return int(self.num_dim * NORM_VEC.shape[0])


class FourierFeaturizerPosCos(IntFeaturizer):
    def __init__(self, num_funcs: int = 9):
        self.num_funcs = num_funcs
        max_freq = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 1
        freqs = 0.5 ** np.linspace(1, max_freq, num_funcs)
        freqs_time_2pi = torch.from_numpy((2 * np.pi * freqs).astype(np.float32))
        super().__init__(embedding_dim=int(freqs_time_2pi.shape[0]))

        combo = torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None] * freqs_time_2pi[None, :]
        self.int_to_feat_matrix = nn.Parameter((-torch.cos(combo) + 1).float(), requires_grad=False)


class FloatFeaturizer(IntFeaturizer):
    def __init__(self):
        super().__init__(embedding_dim=1)
        self.norm_vec = nn.Parameter(torch.from_numpy(NORM_VEC).float(), requires_grad=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tens_shape = tensor.shape
        out_shape = [1] * (len(tens_shape) - 1) + [-1]
        return tensor / self.norm_vec.reshape(*out_shape)

    @property
    def num_dim(self) -> int:
        return 1


def get_embedder(name: str) -> nn.Module:
    if name == "float":
        return FloatFeaturizer()
    if name == "pos-cos":
        return FourierFeaturizerPosCos()
    raise NotImplementedError(f"Unsupported embedder: {name}")

