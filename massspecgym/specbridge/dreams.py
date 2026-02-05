from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DreamsDataFormatA:
    # Used as `x_min` for lin_float_int Fourier features when `fourier_min_freq` is None
    max_tbxic_stdev: float = 1e-4
    max_mz: float = 1000.0


@dataclass
class DreamsConfig:
    """
    Minimal configuration matching the DreaMS pretraining defaults used by SpecBridge.
    """

    n_layers: int = 7
    n_heads: int = 8
    d_fourier: int = 980
    d_peak: int = 44
    d_mz_token: int = 0
    hot_mz_bin_size: float = 0.05
    fourier_strategy: str = "lin_float_int"
    fourier_num_freqs: int | None = None
    fourier_trainable: bool = False
    fourier_min_freq: float | None = None
    dropout: float = 0.1
    att_dropout: float = 0.1
    residual_dropout: float = 0.1
    ff_dropout: float = 0.1
    ff_fourier_depth: int = 5
    ff_fourier_d: int = 512
    ff_peak_depth: int = 1
    no_ffs_bias: bool = False
    no_transformer_bias: bool = True
    pre_norm: bool = True
    scnorm: bool = False
    attn_mech: str = "dot-product"
    graphormer_mz_diffs: bool = True
    graphormer_parametrized: bool = False
    charge_feature: bool = False
    dformat: DreamsDataFormatA = field(default_factory=DreamsDataFormatA)


class FourierFeaturesModule(nn.Module):
    def __init__(
        self,
        strategy: str,
        x_min: float,
        x_max: float,
        trainable: bool = True,
        funcs: str = "both",
        sigma: float = 10.0,
        num_freqs: int = 512,
    ):
        super().__init__()
        from math import ceil

        if strategy not in {"random", "voronov_et_al", "lin_float_int"}:
            raise ValueError(f"Unknown Fourier strategy: {strategy}")
        if funcs not in {"both", "sin", "cos"}:
            raise ValueError(f"Unknown funcs: {funcs}")
        if not (x_min < 1):
            raise ValueError("x_min must be < 1")

        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable

        if strategy == "random":
            b = torch.randn(num_freqs) * sigma
        elif strategy == "voronov_et_al":
            b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2))) for i in range(1, num_freqs)]
            )
        else:  # lin_float_int
            b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)]
                + [1 / (1 * i) for i in range(2, ceil(x_max), 1)]
            )

        self.b = nn.Parameter(b.unsqueeze(0), requires_grad=trainable)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == "both":
            return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        if self.funcs == "cos":
            return torch.cos(x)
        return torch.sin(x)

    def num_features(self) -> int:
        n = int(self.b.shape[1])
        return n if self.funcs != "both" else 2 * n


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
        act_last: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for layer_idx in range(depth):
            d1 = in_dim if layer_idx == 0 else hidden_dim
            d2 = out_dim if layer_idx == depth - 1 else hidden_dim
            layers.append(nn.Linear(d1, d2, bias=bias))
            if layer_idx != depth - 1:
                layers.append(nn.Dropout(p=dropout))
            if layer_idx != depth - 1 or act_last:
                layers.append(nn.ReLU())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class MultiheadAttentionTNQ(nn.Module):
    def __init__(self, cfg: DreamsConfig):
        super().__init__()

        if cfg.d_model % cfg.n_heads != 0:  # type: ignore[attr-defined]
            raise ValueError("Required: d_model % n_heads == 0.")

        self.d_model = cfg.d_model  # type: ignore[attr-defined]
        self.n_heads = cfg.n_heads
        self.dropout = cfg.att_dropout
        self.use_transformer_bias = not cfg.no_transformer_bias
        self.attn_mech = cfg.attn_mech
        self.d_graphormer_params = cfg.d_graphormer_params  # type: ignore[attr-defined]

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim**-0.5

        self.weights = nn.Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        self.biases = nn.Parameter(torch.Tensor(4 * self.d_model)) if self.use_transformer_bias else None

        self.lin_graphormer = (
            nn.Linear(self.d_graphormer_params, self.n_heads, bias=False) if self.d_graphormer_params else None
        )

        mean = 0.0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.0)

    def _proj(self, x, start=0, end=None):
        w = self.weights[start:end, :]
        b = None if self.biases is None else self.biases[start:end]
        return F.linear(x, weight=w, bias=b)

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self._proj(q, end=self.d_model)
            k = self._proj(k, start=self.d_model, end=2 * self.d_model)
            v = self._proj(v, start=2 * self.d_model, end=3 * self.d_model)
        return q, k, v

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv: bool = True):
        bs, n, _ = q.size()

        def split_heads(t):
            return t.reshape(bs, n, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        if self.attn_mech == "dot-product":
            att_weights = torch.einsum("bhnd,bhdm->bhnm", q, k.transpose(-2, -1))
        else:
            raise NotImplementedError(f"{self.attn_mech} not implemented")

        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.lin_graphormer is not None:
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights = att_weights.masked_fill(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)

        out = torch.bmm(att_weights.reshape(-1, n, n), v.reshape(bs * self.n_heads, n, self.head_dim))
        out = out.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        out = self.proj_o(out)
        return out, att_weights


class FeedForwardTNQ(nn.Module):
    def __init__(self, cfg: DreamsConfig):
        super().__init__()
        self.dropout = cfg.ff_dropout
        self.d_model = cfg.d_model  # type: ignore[attr-defined]
        self.ff_dim = 4 * cfg.d_model
        self.use_transformer_bias = not cfg.no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        mean = 0.0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        self._dropout = float(cfg.ff_dropout)

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self._dropout, training=self.training)
        return self.out_proj(y)


class TransformerEncoderTNQ(nn.Module):
    def __init__(self, cfg: DreamsConfig):
        super().__init__()
        self.residual_dropout = cfg.residual_dropout
        self.n_layers = cfg.n_layers
        self.pre_norm = cfg.pre_norm

        self.atts = nn.ModuleList([MultiheadAttentionTNQ(cfg) for _ in range(cfg.n_layers)])
        self.ffs = nn.ModuleList([FeedForwardTNQ(cfg) for _ in range(cfg.n_layers)])

        num_scales = cfg.n_layers * 2 + 1 if cfg.pre_norm else cfg.n_layers * 2
        if cfg.scnorm:
            raise NotImplementedError("ScaleNorm not needed for SpecBridge default config")
        self.scales = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(num_scales)])  # type: ignore[attr-defined]

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        pre_norm = self.pre_norm
        post_norm = not pre_norm

        for i in range(self.n_layers):
            att = self.atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[2 * i]
            ff_scale = self.scales[2 * i + 1]

            residual = x
            x = att_scale(x) if pre_norm else x
            x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            x = att_scale(x) if post_norm else x

            residual = x
            x = ff_scale(x) if pre_norm else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            x = ff_scale(x) if post_norm else x

        x = self.scales[-1](x) if pre_norm else x
        return x


class DreamsEncoder(nn.Module):
    """
    Minimal DreaMS encoder forward producing per-peak embeddings [B, N, d_model].
    """

    def __init__(self, cfg: DreamsConfig):
        super().__init__()

        self.cfg = cfg
        self.d_model = sum(d for d in [cfg.d_fourier, cfg.d_peak, cfg.d_mz_token] if d)
        self.embed_dim = self.d_model

        self.dformat = cfg.dformat
        self.charge_feature = cfg.charge_feature
        self.graphormer_mz_diffs = cfg.graphormer_mz_diffs
        self.d_fourier = cfg.d_fourier

        d_graphormer_params = 0
        if cfg.graphormer_mz_diffs and cfg.graphormer_parametrized:
            d_graphormer_params = cfg.d_fourier if cfg.d_fourier else 1

        # attach derived fields so TNQ modules match ckpt layout
        cfg_local = cfg
        cfg_local.d_model = self.d_model  # type: ignore[attr-defined]
        cfg_local.d_graphormer_params = d_graphormer_params  # type: ignore[attr-defined]

        token_dim = 2 + (1 if cfg.charge_feature else 0)
        self.ff_peak = FeedForwardModule(
            in_dim=token_dim,
            out_dim=cfg.d_peak,
            hidden_dim=cfg.d_peak,
            depth=cfg.ff_peak_depth,
            dropout=cfg.dropout,
            bias=not cfg.no_ffs_bias,
        )

        if not cfg.d_fourier:
            raise NotImplementedError("SpecBridge defaults expect Fourier features enabled.")

        self.fourier_enc = FourierFeaturesModule(
            strategy=cfg.fourier_strategy,
            num_freqs=(cfg.fourier_num_freqs or 512),
            x_min=(cfg.fourier_min_freq if cfg.fourier_min_freq is not None else cfg.dformat.max_tbxic_stdev),
            x_max=cfg.dformat.max_mz,
            trainable=cfg.fourier_trainable,
        )
        self.ff_fourier = FeedForwardModule(
            in_dim=self.fourier_enc.num_features(),
            out_dim=cfg.d_fourier,
            hidden_dim=cfg.ff_fourier_d,
            depth=cfg.ff_fourier_depth,
            dropout=cfg.dropout,
            bias=not cfg.no_ffs_bias,
        )

        self.transformer_encoder = TransformerEncoderTNQ(cfg_local)
    def _normalize_spec(self, spec):
        return spec / torch.tensor([self.dformat.max_mz, 1.0], device=spec.device, dtype=spec.dtype)

    def forward(self, spec, charge=None):
        padding_mask = spec[:, :, 0] == 0
        if self.charge_feature:
            raise NotImplementedError("charge_feature not supported in SpecBridge defaults")

        peak_embs = self.ff_peak(self._normalize_spec(spec))
        fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
        x = torch.cat([peak_embs, fourier_features], dim=-1)

        graphormer_dists = None
        if self.graphormer_mz_diffs:
            graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)

        return self.transformer_encoder(x, padding_mask, graphormer_dists)
