#!/bin/bash
#SBATCH --job-name=mist_fps
#SBATCH --partition=mit_normal_gpu,pi_ccoley,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mist_fps_%j.out
#SBATCH --error=logs/mist_fps_%j.err
#SBATCH --requeue

echo "=========================================="
echo "MIST Fingerprint Prediction (all 5 ranks + GT)"
echo "Job: $SLURM_JOB_ID"
echo "=========================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5

cd /home/mrunali/MassSpecGym

SCRATCH=/home/mrunali/orcd/scratch/msg_benchmark
MSG=/home/mrunali/DiffMS/msg

python scripts/predict_mist_fingerprints.py \
    --mist-ckpt checkpoints/encoder_msg.pt \
    --subform-base $SCRATCH/subformulae \
    --labels $MSG/labels.tsv \
    --split $MSG/split.tsv \
    --output-dir $SCRATCH/fingerprints \
    --max-k 5 \
    --accelerator gpu

echo "Done: $SLURM_JOB_ID"
