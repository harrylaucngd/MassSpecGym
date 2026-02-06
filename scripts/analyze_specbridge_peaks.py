"""
Analyze MassSpecGym test-set peak-count distribution and SpecBridge retrieval performance vs. peak count.

This script uses the same spectrum preprocessing as `eval_specbridge_retrieval.py` (SpecTokenizer),
so the peak count is computed *after* matchms filtering and peak limiting, and excludes padding.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from massspecgym.data import MassSpecDataModule, RetrievalDataset
from massspecgym.data.transforms import SpecTokenizer
from massspecgym.models.base import Stage
from massspecgym.models.retrieval import SpecBridgeRetrieval


@dataclass(frozen=True)
class BucketStats:
    bucket: str
    n: int
    hit_rate_at_1: float
    hit_rate_at_5: float
    hit_rate_at_20: float


def _count_ms2_peaks_from_tokenized(spec: torch.Tensor) -> int:
    """
    Count non-padding MS2 peaks from SpecTokenizer output.

    SpecTokenizer output contains an optional precursor row at index 0 with intensity > 0,
    and padding rows with intensity == 0. We exclude the precursor row.
    """
    if spec.ndim != 2 or spec.shape[-1] != 2:
        raise ValueError(f"Unexpected spec shape: {tuple(spec.shape)} (expected [N,2]).")

    inten = spec[:, 1]
    nonzero = int((inten > 0).sum().item())
    # subtract precursor row if present (it always has intensity > 0 in SpecTokenizer)
    return max(nonzero - 1, 0)


def _hit_rate_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if labels.numel() == 0:
        return float("nan")
    kk = min(int(k), int(scores.numel()))
    topk = torch.topk(scores, k=kk, largest=True).indices
    return float(labels[topk].any().item())


def _bucket_edges_from_arg(arg: str) -> list[int]:
    """
    Parse bucket edges like: "0,1,2,3,5,10,20,40,60" (must be sorted ascending).
    """
    edges = [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise ValueError("Need at least 2 bucket edges.")
    if edges != sorted(edges):
        raise ValueError("Bucket edges must be sorted ascending.")
    return edges


def _bucket_label(lo: int, hi: int) -> str:
    return f"[{lo},{hi})"


def _assign_bucket(x: int, edges: list[int]) -> str:
    # last edge is exclusive upper bound; put x==edges[-1] into the last bucket by extending upper bound by 1.
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= x < hi:
            return _bucket_label(lo, hi)
    if x >= edges[-1]:
        return f"[{edges[-1]},inf)"
    return f"(-inf,{edges[0]})"  # unlikely for non-negative x


def _plot_hist(values: Iterable[int], out_png: Path, title: str, xlabel: str) -> None:
    vals = np.asarray(list(values), dtype=np.int32)
    plt.figure(figsize=(7, 4))
    if vals.size == 0:
        raise ValueError("No values to plot.")
    # Discrete integer bins centered on integers.
    bins = np.arange(vals.min(), vals.max() + 2) - 0.5
    plt.hist(vals, bins=bins, color="#2b8cbe", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_bucket_metric(df: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    x = np.arange(len(df))
    plt.plot(x, df["hit_rate@1"], marker="o", label="HitRate@1")
    plt.plot(x, df["hit_rate@5"], marker="o", label="HitRate@5")
    plt.plot(x, df["hit_rate@20"], marker="o", label="HitRate@20")
    plt.xticks(x, df["bucket"], rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("MS2 peak count bucket")
    plt.ylabel("Mean hit rate")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def _plot_count_metric(df: pd.DataFrame, out_png: Path, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    x = df["ms2_peaks"].to_numpy()
    plt.plot(x, df["hit_rate@1"], marker="o", label="HitRate@1")
    plt.plot(x, df["hit_rate@5"], marker="o", label="HitRate@5")
    plt.plot(x, df["hit_rate@20"], marker="o", label="HitRate@20")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("MS2 peak count")
    plt.ylabel("Mean hit rate")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specbridge-ckpt", type=Path, required=True)
    ap.add_argument("--candidate-embeddings", type=Path, required=True)
    ap.add_argument("--candidates-pth", type=str, default=None)
    ap.add_argument("--dataset-pth", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/specbridge_peak_analysis"))

    ap.add_argument("--n-peaks", type=int, default=60)
    ap.add_argument("--max-mz", type=int, default=1000)
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)

    ap.add_argument("--accelerator", type=str, default="cpu")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--bucket-edges", type=str, default="0,1,2,3,5,10,20,30,40,60,100")
    ap.add_argument("--skip-mces", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true", help="Only compute/plot peak-count distribution (no model inference).")
    ap.add_argument("--min-count-per-peak", type=int, default=10, help="Min samples for per-peak-count aggregation.")
    ap.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
    )
    args = ap.parse_args()

    if torch.cuda.is_available() and args.accelerator in {"gpu", "cuda"}:
        torch.set_float32_matmul_precision(args.matmul_precision)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_transform = SpecTokenizer(
        n_peaks=args.n_peaks,
        matchms_kwargs=dict(mz_to=args.max_mz),
    )
    dataset = RetrievalDataset(
        pth=args.dataset_pth,
        spec_transform=spec_transform,
        mol_transform=None,
        candidates_pth=args.candidates_pth,
        return_mol_freq=False,
        return_identifier=True,
    )
    dm = MassSpecDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    dm.setup(stage="test")

    device = torch.device("cuda" if (args.accelerator in {"gpu", "cuda"} and torch.cuda.is_available()) else "cpu")
    model = None
    if not args.skip_retrieval:
        # Instantiate model (pure inference)
        no_mces = (Stage.VAL, Stage.TEST) if args.skip_mces else (Stage.VAL,)
        model = SpecBridgeRetrieval(
            specbridge_ckpt_path=args.specbridge_ckpt,
            candidate_embeddings_path=args.candidate_embeddings,
            normalize=args.normalize,
            df_test_path=None,
            log_only_loss_at_stages=(),
            no_mces_metrics_at_stages=no_mces,
        ).to(device)
        model.eval()

    # Iterate test loader manually to collect per-sample stats
    loader = dm.test_dataloader()
    per_sample_rows = []
    all_peak_counts = []

    edges = _bucket_edges_from_arg(args.bucket_edges)

    with torch.no_grad():
        for batch in loader:
            # batch["spec"]: [B, N, 2]
            spec = batch["spec"]
            batch_ptr = batch["batch_ptr"]
            labels = batch["labels"]

            # compute peak counts for each sample in batch
            peak_counts = [_count_ms2_peaks_from_tokenized(s) for s in spec]
            all_peak_counts.extend(peak_counts)

            if model is not None:
                # model step
                batch_on_device = dict(batch)
                batch_on_device["spec"] = batch_on_device["spec"].to(device=device, non_blocking=True)
                batch_on_device["batch_ptr"] = batch_on_device["batch_ptr"].to(device=device, non_blocking=True)
                if isinstance(batch_on_device.get("labels"), torch.Tensor):
                    batch_on_device["labels"] = batch_on_device["labels"].to(device=device, non_blocking=True)

                out = model.step(batch_on_device)
                scores = out["scores"].detach().cpu()
                labels_cpu = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
                batch_ptr_cpu = batch_ptr.detach().cpu()

                # unbatch scores/labels using batch_ptr
                offsets = torch.cumsum(batch_ptr_cpu, dim=0)
                starts = torch.cat([torch.zeros(1, dtype=offsets.dtype), offsets[:-1]])

                identifiers = batch.get("identifier")
                if identifiers is None:
                    identifiers = [str(i) for i in range(int(batch_ptr_cpu.shape[0]))]

                for idx, (s, e, pk) in enumerate(zip(starts.tolist(), offsets.tolist(), peak_counts)):
                    scores_i = scores[s:e]
                    labels_i = labels_cpu[s:e].to(dtype=torch.bool)
                    per_sample_rows.append(
                        {
                            "identifier": str(identifiers[idx]),
                            "ms2_peaks": int(pk),
                            "bucket": _assign_bucket(int(pk), edges),
                            "hit@1": _hit_rate_at_k(scores_i, labels_i, 1),
                            "hit@5": _hit_rate_at_k(scores_i, labels_i, 5),
                            "hit@20": _hit_rate_at_k(scores_i, labels_i, 20),
                        }
                    )

    # Peak-count distribution plot
    _plot_hist(
        all_peak_counts,
        out_png=out_dir / "test_ms2_peak_count_hist.png",
        title="MassSpecGym test set MS2 peak count distribution (post-preprocessing)",
        xlabel="MS2 peak count",
    )

    wrote = [out_dir / "test_ms2_peak_count_hist.png"]

    if model is not None:
        # Aggregate performance by bucket
        df = pd.DataFrame(per_sample_rows)
        df.to_csv(out_dir / "per_sample_hits.csv", index=False)
        wrote.append(out_dir / "per_sample_hits.csv")

        grouped = (
            df.groupby("bucket", sort=False)
            .agg(
                n=("identifier", "count"),
                hit_rate1=("hit@1", "mean"),
                hit_rate5=("hit@5", "mean"),
                hit_rate20=("hit@20", "mean"),
            )
            .reset_index()
            .rename(columns={"hit_rate1": "hit_rate@1", "hit_rate5": "hit_rate@5", "hit_rate20": "hit_rate@20"})
        )
        grouped.to_csv(out_dir / "hit_rates_by_peak_bucket.csv", index=False)
        wrote.append(out_dir / "hit_rates_by_peak_bucket.csv")

        _plot_bucket_metric(
            grouped,
            out_png=out_dir / "hit_rates_by_peak_bucket.png",
            title="SpecBridge retrieval performance vs MS2 peak count (bucketed)",
        )
        wrote.append(out_dir / "hit_rates_by_peak_bucket.png")

        # Also aggregate by exact peak count (with minimum support)
        by_count = (
            df.groupby("ms2_peaks", sort=True)
            .agg(
                n=("identifier", "count"),
                hit_rate1=("hit@1", "mean"),
                hit_rate5=("hit@5", "mean"),
                hit_rate20=("hit@20", "mean"),
            )
            .reset_index()
            .rename(columns={"hit_rate1": "hit_rate@1", "hit_rate5": "hit_rate@5", "hit_rate20": "hit_rate@20"})
        )
        by_count = by_count[by_count["n"] >= int(args.min_count_per_peak)]
        by_count.to_csv(out_dir / "hit_rates_by_peak_count.csv", index=False)
        wrote.append(out_dir / "hit_rates_by_peak_count.csv")

        _plot_count_metric(
            by_count,
            out_png=out_dir / "hit_rates_by_peak_count.png",
            title=f"SpecBridge retrieval performance vs MS2 peak count (n>={int(args.min_count_per_peak)})",
        )
        wrote.append(out_dir / "hit_rates_by_peak_count.png")

    print("Wrote:")
    for p in wrote:
        print(f"- {p}")


if __name__ == "__main__":
    main()
