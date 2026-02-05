"""
Evaluate SpecBridge retrieval (standard or formula-bonus) using MassSpecGym's official metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from massspecgym.data import MassSpecDataModule, RetrievalDataset
from massspecgym.data.transforms import SpecTokenizer
from massspecgym.models.retrieval import SpecBridgeRetrieval
from massspecgym.models.base import Stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specbridge-ckpt", type=Path, required=True, help="Path to SpecBridge adapter checkpoint (e.g., checkpoints/last.pt).")
    ap.add_argument("--candidate-embeddings", type=Path, required=True, help="Path to SmilesEmbeddingStore created by specbridge_precompute_candidate_embeddings.py.")
    ap.add_argument("--candidates-pth", type=str, default=None, help="Candidates JSON path or 'bonus' for formula-bonus candidates.")
    ap.add_argument("--dataset-pth", type=str, default=None, help="Optional MassSpecGym.tsv/.mgf path (downloads from HF if omitted).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accelerator", type=str, default="cpu")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--n-peaks", type=int, default=60)
    ap.add_argument("--max-mz", type=int, default=1000)
    ap.add_argument("--normalize", action="store_true", help="Use cosine similarity (recommended; matches SpecBridge paper).")
    ap.add_argument("--df-test-pth", type=Path, default=None, help="Optional path to store per-sample test dataframe (*.pkl).")
    ap.add_argument("--skip-mces", action="store_true", help="Skip MCES@1 computation (much faster; keeps HitRate@K).")
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

    spec_transform = SpecTokenizer(
        n_peaks=args.n_peaks,
        matchms_kwargs=dict(mz_to=args.max_mz),
    )
    dataset = RetrievalDataset(
        pth=args.dataset_pth,
        spec_transform=spec_transform,
        mol_transform=None,  # keep candidates as SMILES; SpecBridge model uses embedding store
        candidates_pth=args.candidates_pth,
        return_mol_freq=False,
        return_identifier=True,
    )
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    no_mces_metrics_at_stages = (Stage.VAL,) if not args.skip_mces else (Stage.VAL, Stage.TEST)
    model = SpecBridgeRetrieval(
        specbridge_ckpt_path=args.specbridge_ckpt,
        candidate_embeddings_path=args.candidate_embeddings,
        normalize=args.normalize,
        df_test_path=args.df_test_pth,
        log_only_loss_at_stages=(),
        no_mces_metrics_at_stages=no_mces_metrics_at_stages,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
