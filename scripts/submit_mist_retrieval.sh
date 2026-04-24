#!/bin/bash
#SBATCH --job-name=mist_retr
#SBATCH --partition=mit_normal_gpu,pi_ccoley,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mist_retr_%j.out
#SBATCH --error=logs/mist_retr_%j.err

# Usage: sbatch scripts/submit_mist_retrieval.sh [gt|mistcf] [mass|bonus]
# Defaults: gt mass

SUBFORM_TYPE=${1:-gt}
CAND_TYPE=${2:-mass}

if [ "$SUBFORM_TYPE" = "gt" ]; then
    SUBFORM_FOLDER="data/msg/subformulae/default_subformulae"
    RUN_LABEL="gt"
elif [ "$SUBFORM_TYPE" = "mistcf" ]; then
    SUBFORM_FOLDER="data/msg/subformulae/mistcf_top1_subformulae"
    RUN_LABEL="mistcf_top1"
else
    echo "Unknown subform type: $SUBFORM_TYPE (use gt or mistcf)"
    exit 1
fi

CAND_ARGS=""
if [ "$CAND_TYPE" = "bonus" ]; then
    CAND_ARGS="--candidates-pth bonus"
    RUN_LABEL="${RUN_LABEL}_bonus"
fi

echo "=========================================="
echo "MIST Retrieval: subform=$SUBFORM_TYPE, candidates=$CAND_TYPE"
echo "Subform folder: $SUBFORM_FOLDER"
echo "Label: $RUN_LABEL"
echo "=========================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5

cd /home/mrunali/MassSpecGym
mkdir -p logs results

python scripts/eval_mist_msg_retrieval.py \
    --mist-ckpt checkpoints/encoder_msg.pt \
    --labels-pth data/msg/labels.tsv \
    --split-pth data/msg/split.tsv \
    --subform-folder "$SUBFORM_FOLDER" \
    $CAND_ARGS \
    --accelerator gpu \
    --batch-size 16 \
    --num-workers 16 \
    --pin-memory \
    --skip-mces \
    2>&1 | tee "results/mist_retrieval_${RUN_LABEL}.log"

echo "Done: $RUN_LABEL"
