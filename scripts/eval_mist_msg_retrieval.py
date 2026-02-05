"""
Evaluate DiffMS/MIST encoder retrieval on MSG subformula-assigned MassSpecGym splits
using MassSpecGym's official retrieval metrics (HitRate@K, optional MCES@1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from massspecgym.data import MassSpecDataModule
from massspecgym.mist import MsgMistRetrievalDataset
from massspecgym.models.base import Stage
from massspecgym.models.retrieval import MistEncoderRetrieval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mist-ckpt", type=Path, required=True, help="Path to the MIST encoder checkpoint (e.g. checkpoints/encoder_msg.pt).")
    ap.add_argument("--labels-pth", type=Path, required=True, help="Path to MSG labels.tsv (must contain spec, smiles, instrument).")
    ap.add_argument("--split-pth", type=Path, required=True, help="Path to MSG split.tsv (columns: spec, fold).")
    ap.add_argument("--subform-folder", type=Path, required=True, help="Folder with subformula JSON trees (<spec>.json).")
    ap.add_argument("--candidates-pth", type=str, default=None, help="Candidates JSON path or 'bonus' for formula-bonus candidates.")
    ap.add_argument("--threshold", type=float, default=0.187, help="MIST fingerprint threshold (default: 0.187).")

    ap.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size (encoder inference still runs per-spectrum with B=1 internally).")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)

    ap.add_argument("--accelerator", type=str, default="cpu")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--skip-mces", action="store_true", help="Skip MCES@1 computation (much faster; keeps HitRate@K).")
    ap.add_argument("--sanity-shuffle-candidates", action="store_true", help="Shuffle candidate fingerprints within each query (should drop to ~random).")
    ap.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="torch.set_float32_matmul_precision setting (GPU speed/accuracy trade-off).",
    )
    args = ap.parse_args()

    if torch.cuda.is_available() and args.accelerator in {"gpu", "cuda"}:
        torch.set_float32_matmul_precision(args.matmul_precision)

    dataset = MsgMistRetrievalDataset(
        labels_pth=args.labels_pth,
        split_pth=args.split_pth,
        subform_folder=args.subform_folder,
        candidates_pth=args.candidates_pth,
        cls_mode="ms1",
    )
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    no_mces_metrics_at_stages = (Stage.VAL,) if not args.skip_mces else (Stage.VAL, Stage.TEST)
    model = MistEncoderRetrieval(
        mist_ckpt_path=args.mist_ckpt,
        threshold=args.threshold,
        sanity_shuffle_candidates=args.sanity_shuffle_candidates,
        df_test_path=None,
        log_only_loss_at_stages=(),
        no_mces_metrics_at_stages=no_mces_metrics_at_stages,
    )
    print(f"[MIST] Encoder kwargs: {getattr(model, 'encoder_kwargs', None)}")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
