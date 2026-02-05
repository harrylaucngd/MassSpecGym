"""
Precompute ChemBERTa embeddings for the MassSpecGym retrieval candidate bank.

This is intended for SpecBridge retrieval evaluation inside MassSpecGym.
It produces a `SmilesEmbeddingStore` file that can be consumed by
`massspecgym.models.retrieval.SpecBridgeRetrieval`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import massspecgym.utils as utils
from massspecgym.specbridge.checkpoint import load_specbridge_checkpoint
from massspecgym.specbridge.chemberta import ChemBertaEmbedder
from massspecgym.specbridge.embedding_store import EmbeddingStoreMetadata, SmilesEmbeddingStore


def _resolve_candidates_path(candidates_pth: str | None) -> Path:
    if candidates_pth is None:
        return Path(
            utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_mass.json")
        )
    if candidates_pth == "bonus":
        return Path(
            utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json")
        )
    p = Path(candidates_pth)
    if p.is_file():
        return p
    return Path(utils.hugging_face_download(candidates_pth))


def _resolve_dataset_tsv(dataset_pth: str | None) -> Path:
    if dataset_pth is None:
        return Path(utils.hugging_face_download("MassSpecGym.tsv"))
    return Path(dataset_pth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specbridge-ckpt", type=Path, required=True, help="Path to SpecBridge adapter checkpoint (e.g., checkpoints/last.pt).")
    ap.add_argument("--output", type=Path, required=True, help="Output path for embedding store (*.pt).")
    ap.add_argument("--candidates-pth", type=str, default=None, help="Candidates JSON path or 'bonus' for formula-bonus candidates.")
    ap.add_argument("--dataset-pth", type=str, default=None, help="Path to MassSpecGym.tsv (optional; downloads from HF if omitted).")
    ap.add_argument("--fold", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--chemberta-model", type=str, default=None, help="HF model name. If omitted, inferred from SpecBridge ckpt args.")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--normalize", action="store_true", help="Store L2-normalized embeddings (recommended).")
    ap.add_argument("--load-weights-from-ckpt", action="store_true", help="Load ChemBERTa weights from SpecBridge ckpt state_dict (recommended for exact reproducibility).")
    args = ap.parse_args()

    ckpt = load_specbridge_checkpoint(args.specbridge_ckpt)
    chemberta_model = args.chemberta_model or ckpt.args.get("chemberta_model") or "seyonec/ChemBERTa-zinc-base-v1"

    candidates_path = _resolve_candidates_path(args.candidates_pth)
    dataset_path = _resolve_dataset_tsv(args.dataset_pth)

    df = pd.read_csv(dataset_path, sep="\t", usecols=["smiles", "fold"])
    if args.fold != "all":
        df = df[df["fold"] == args.fold]

    with open(candidates_path, "r") as f:
        cand_map = json.load(f)

    smiles_set: set[str] = set()
    missing = 0
    for sm in df["smiles"].tolist():
        cands = cand_map.get(sm)
        if not cands:
            missing += 1
            continue
        smiles_set.update(cands)
    smiles = sorted(smiles_set)
    if len(smiles) == 0:
        raise RuntimeError("No candidate SMILES collected; check fold/candidates inputs.")

    print(f"[specbridge] candidates_file={candidates_path}")
    print(f"[specbridge] fold={args.fold} | queries={len(df)} | missing_candidates={missing}")
    print(f"[specbridge] unique_candidate_smiles={len(smiles)}")
    print(f"[specbridge] chemberta_model={chemberta_model}")

    embedder = ChemBertaEmbedder(
        model_name=str(chemberta_model),
        max_length=args.max_length,
        device=args.device,
        dtype=None,
    )
    if args.load_weights_from_ckpt:
        embedder.load_weights_from_state_dict(ckpt.model_state, prefix="chem_mdl")

    embs = []
    for i in range(0, len(smiles), args.batch_size):
        chunk = smiles[i : i + args.batch_size]
        z = embedder.embed(chunk, batch_size=args.batch_size)
        if args.normalize:
            z = torch.nn.functional.normalize(z, dim=-1)
        z = z.to(dtype=getattr(torch, args.dtype)).cpu()
        embs.append(z)
        if (i // args.batch_size) % 50 == 0:
            print(f"[specbridge] embedded {min(i + args.batch_size, len(smiles))}/{len(smiles)}")

    embeddings = torch.cat(embs, dim=0)
    md = EmbeddingStoreMetadata(
        model_name=str(chemberta_model),
        normalized=bool(args.normalize),
        dtype=str(embeddings.dtype).replace("torch.", ""),
        dim=int(embeddings.shape[1]),
    )
    SmilesEmbeddingStore.save(args.output, smiles=smiles, embeddings=embeddings, metadata=md)
    print(f"[specbridge] wrote {args.output} | shape={tuple(embeddings.shape)} | normalized={md.normalized}")


if __name__ == "__main__":
    main()

