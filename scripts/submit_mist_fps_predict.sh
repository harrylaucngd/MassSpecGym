#!/bin/bash
#SBATCH --job-name=mist_fps_predict
#SBATCH --partition=pi_ccoley,mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mist_fps_predict_%j.out
#SBATCH --error=logs/mist_fps_predict_%j.err

echo "Job: $SLURM_JOB_ID"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym

python scripts/predict_mist_fingerprints.py \
    --mist-ckpt checkpoints/encoder_msg.pt \
    --subform-base data/msg/subformulae \
    --labels data/msg/labels.tsv \
    --split data/msg/split.tsv \
    --output-dir results/mist_fps \
    --max-k 1 \
    --also-gt \
    --accelerator gpu

echo "Done"
