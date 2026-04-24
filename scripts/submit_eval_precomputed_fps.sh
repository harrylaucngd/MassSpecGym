#!/bin/bash
#SBATCH --job-name=eval_fps
#SBATCH --partition=pi_ccoley,ou_cheme,mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_fps_%j.out
#SBATCH --error=logs/eval_fps_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5

cd /home/mrunali/MassSpecGym
mkdir -p logs

python scripts/eval_precomputed_fps.py \
    --scratch-dir /home/mrunali/orcd/scratch/msg_benchmark \
    --labels /home/mrunali/orcd/scratch/msg_benchmark/labels.tsv \
    --split /home/mrunali/orcd/scratch/msg_benchmark/split.tsv \
    --candidates ~/ms-data/data/msg/MassSpecGym_retrieval_candidates_mass.json \
    --modes top1 top5_avg top5_max top5_rank_avg \
    --max-ranks 5 \
    --hits 1 5 10 20 \
    2>&1 | tee results/eval_precomputed_fps.log

echo "Done"
