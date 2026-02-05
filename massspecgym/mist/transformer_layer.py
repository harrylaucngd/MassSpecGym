from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, Parameter
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class TransformerEncoderLayer(Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        additive_attn: bool = False,
        pairwise_featurization: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.pairwise_featurization = pairwise_featurization
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            additive_attn=additive_attn,
            pairwise_featurization=self.pairwise_featurization,
            **factory_kwargs,
        )
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        src: Tensor,
        pairwise_features: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), pairwise_features, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, pairwise_features, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x, pairwise_features

    def _sa_block(self, x: Tensor, pairwise_features: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, pairwise_features=pairwise_features)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        additive_attn: bool = False,
        pairwise_featurization: bool = False,
        dropout: float = 0.0,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True
        self.additive_attn = additive_attn
        self.pairwise_featurization = pairwise_featurization

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        if self.additive_attn:
            head_1_input = self.head_dim * (3 if self.pairwise_featurization else 2)
            self.attn_weight_1_weight = Parameter(torch.empty((self.num_heads, head_1_input, self.head_dim), **factory_kwargs))
            self.attn_weight_1_bias = Parameter(torch.empty((self.num_heads, self.head_dim), **factory_kwargs))
            self.attn_weight_2_weight = Parameter(torch.empty((self.num_heads, self.head_dim, 1), **factory_kwargs))
            self.attn_weight_2_bias = Parameter(torch.empty((self.num_heads, 1), **factory_kwargs))
        else:
            if self.pairwise_featurization:
                self.bias_u = Parameter(torch.empty((self.num_heads, self.head_dim), **factory_kwargs))
                self.bias_v = Parameter(torch.empty((self.num_heads, self.head_dim), **factory_kwargs))

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)
        if self.additive_attn:
            xavier_uniform_(self.attn_weight_1_weight)
            constant_(self.attn_weight_1_bias, 0.0)
            xavier_uniform_(self.attn_weight_2_weight)
            constant_(self.attn_weight_2_bias, 0.0)
        else:
            if self.pairwise_featurization:
                xavier_uniform_(self.bias_u)
                xavier_uniform_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        pairwise_features: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # This implementation is sufficient for loading the MSG encoder checkpoint and running inference.
        # It mirrors DiffMS/src/mist/models/transformer_layer.py behavior for dot-product attention.
        if self.batch_first:
            # convert to (seq, batch, embed)
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        tgt_len, bsz, _ = query.shape
        head_dim = self.head_dim
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        q = q * (1.0 / math.sqrt(head_dim))
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] True for padded
            mask = key_padding_mask.repeat_interleave(self.num_heads, dim=0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if not need_weights:
            return attn_output, None

        # Return average attention weights if requested.
        attn_weights_ = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
        if average_attn_weights:
            attn_weights_ = attn_weights_.mean(dim=1)
        return attn_output, attn_weights_

