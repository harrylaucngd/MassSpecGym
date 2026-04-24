#!/bin/bash
#SBATCH --job-name=mist_eval
#SBATCH --partition=mit_preemptable,pi_ccoley
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/mist_eval_%j.out
#SBATCH --error=logs/mist_eval_%j.err
#SBATCH --requeue

echo "=========================================="
echo "MIST Retrieval Evaluation from Pre-computed FPs"
echo "Job: $SLURM_JOB_ID"
echo "=========================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5

cd /home/mrunali/MassSpecGym

SCRATCH=/home/mrunali/orcd/scratch/msg_benchmark
MSG=/home/mrunali/DiffMS/msg

python scripts/eval_retrieval_from_fps.py \
    --fp-dir $SCRATCH/fingerprints \
    --labels $MSG/labels.tsv \
    --split $MSG/split.tsv \
    --candidates-json /home/mrunali/ms-data/data/msg/MassSpecGym_retrieval_candidates_mass.json \
    --output-dir $SCRATCH/results

echo "Done: $SLURM_JOB_ID"
