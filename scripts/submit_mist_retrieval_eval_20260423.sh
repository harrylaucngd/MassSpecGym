#!/bin/bash
#SBATCH --job-name=mist_retr_eval
#SBATCH --partition=pi_ccoley,mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mist_retr_eval_%j.out
#SBATCH --error=logs/mist_retr_eval_%j.err

# Evaluate MISTFingerprintRetrieval via scripts/run.py using encoder_msg.pt.
# Supports gt (default_subformulae) and mistcf_rank{1..4} subformula sets,
# and mass or bonus (formula) candidate sets.
#
# Usage:
#   sbatch scripts/submit_mist_retrieval_eval_20260423.sh [gt|mistcf_rank1|...] [mass|bonus] [cosine|tanimoto]
#
# Defaults: gt bonus cosine
#
# Spectrum/metadata TSV: omit --dataset_pth to download MassSpecGym.tsv from HuggingFace,
# or set MASMSG_TSV to a local path (full TSV with mzs/intensities, not MIST labels.tsv).
#
# MCES@1 is slow (ILP per spectrum); --skip_mces_test keeps hit-rate metrics. Remove the flag
# for full benchmark MCES@1.

SUBFORM_TYPE=${1:-gt}
CAND_TYPE=${2:-bonus}
SIMILARITY=${3:-cosine}

case "$SUBFORM_TYPE" in
    gt)             SUBFORM_FOLDER="data/msg/subformulae/default_subformulae" ;;
    mistcf_rank*)   SUBFORM_FOLDER="data/msg/subformulae/${SUBFORM_TYPE}_subformulae" ;;
    *)  echo "Unknown subform type: $SUBFORM_TYPE (use gt or mistcf_rank{1..4})"; exit 1 ;;
esac

case "$CAND_TYPE" in
    mass)   CANDS_PTH=None ;;
    bonus)  CANDS_PTH=bonus ;;
    *)  echo "Unknown candidate type: $CAND_TYPE (use mass or bonus)"; exit 1 ;;
esac

RUN_LABEL="mist_retr_${SUBFORM_TYPE}_${CAND_TYPE}_${SIMILARITY}"

echo "=================================================="
echo "MIST Retrieval Eval via run.py"
echo "  subform: $SUBFORM_TYPE  ($SUBFORM_FOLDER)"
echo "  candidates: $CAND_TYPE"
echo "  run label: $RUN_LABEL"
echo "  job: $SLURM_JOB_ID"
echo "=================================================="

source ~/miniforge3/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym
mkdir -p logs

python scripts/run.py \
    --job_key  "$RUN_LABEL" \
    --run_name "$RUN_LABEL" \
    --no_wandb \
    --test_only \
    --skip_mces_test \
    --task retrieval \
    --model mist_fingerprint \
    --subform_folder "$SUBFORM_FOLDER" \
    --encoder_checkpoint checkpoints/encoder_msg.pt \
    --candidates_pth "$CANDS_PTH" \
    ${MASMSG_TSV:+--dataset_pth "$MASMSG_TSV"} \
    --split_pth data/msg/split.tsv \
    --fp_similarity "$SIMILARITY" \
    --fp_save_path "results/mist_fps/fingerprints_${RUN_LABEL}_raw_$(date +%Y%m%d).pt" \
    --fp_size 4096 \
    --accelerator gpu \
    --devices 1 \
    --batch_size 32 \
    --num_workers 8 && echo "Done: $RUN_LABEL"
