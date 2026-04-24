#!/bin/bash
#SBATCH --job-name=fps_eval_diff
#SBATCH --partition=pi_ccoley,ou_cheme,ou_cheme_preemptable,mit_preemptable
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fps_eval_diff_%j.out
#SBATCH --error=logs/fps_eval_diff_%j.err

# Cross-check eval_retrieval_from_fps.py vs trainer.test for job 12427444.
# trainer.test (job 12427444) reported (tanimoto + bonus/formula cands, continuous fps):
#   HitRate@1  = 0.4782
#   HitRate@5  = 0.5739
#   HitRate@20 = 0.6852
#
# This script runs eval_retrieval_from_fps.py three ways on the SAME .pt file:
#   A) tanimoto, continuous (should match trainer.test)
#   B) tanimoto, thresholded at 0.187 (rounding experiment)
#   C) cosine, continuous (sanity; trainer.test used tanimoto)

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym

FP_PT="results/mist_fps/fingerprints_mist_retr_gt_bonus_tanimoto_raw_20260423.pt"
if [ ! -f "$FP_PT" ]; then
    echo "ERROR: $FP_PT not found"; exit 1
fi

# Stage the single .pt into a dedicated dir (script globs fingerprints_*.pt)
FP_DIR="results/mist_fps/eval_12427444"
mkdir -p "$FP_DIR"
cp -n "$FP_PT" "$FP_DIR/fingerprints_mist_retr_gt_bonus_tanimoto_raw_20260423.pt"

CANDS_JSON="$(python -c 'import massspecgym.utils as u; print(u.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json"))')"
echo "Using candidates: $CANDS_JSON"

LABELS="data/msg/labels.tsv"
SPLIT="data/msg/split.tsv"
OUT="results/mist_fps/eval_12427444"
# Shared cache of Morgan fps + inchikeys keyed by SMILES. First run populates
# (~20 min of RDKit work); subsequent runs load in seconds.
FP_CACHE="results/mist_fps/morgan_fp_cache_formula_4096.npz"

echo "=========================================="
echo "[A] tanimoto, continuous (matches trainer.test)"
echo "=========================================="
python scripts/eval_retrieval_from_fps.py \
    --fp-dir "$FP_DIR" --labels "$LABELS" --split "$SPLIT" \
    --candidates-json "$CANDS_JSON" --output-dir "$OUT" \
    --fp-cache "$FP_CACHE" \
    --similarity tanimoto

echo ""
echo "=========================================="
echo "[B] tanimoto, thresholded at 0.187"
echo "=========================================="
python scripts/eval_retrieval_from_fps.py \
    --fp-dir "$FP_DIR" --labels "$LABELS" --split "$SPLIT" \
    --candidates-json "$CANDS_JSON" --output-dir "$OUT" \
    --fp-cache "$FP_CACHE" \
    --similarity tanimoto --threshold 0.187

echo ""
echo "=========================================="
echo "[C] cosine, continuous (sanity)"
echo "=========================================="
python scripts/eval_retrieval_from_fps.py \
    --fp-dir "$FP_DIR" --labels "$LABELS" --split "$SPLIT" \
    --candidates-json "$CANDS_JSON" --output-dir "$OUT" \
    --fp-cache "$FP_CACHE" \
    --similarity cosine

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "trainer.test (job 12427444, tanimoto, bonus):"
echo "  HitRate@1  = 0.4782"
echo "  HitRate@5  = 0.5739"
echo "  HitRate@20 = 0.6852"
echo ""
for f in "$OUT"/retrieval_results_*.json; do
    echo "--- $f ---"
    python -c "import json; d=json.load(open('$f')); print('similarity=', d['similarity'], 'threshold=', d['threshold']); [print(f'  {k}: {v}') for k,v in d['results'].items()]"
done
