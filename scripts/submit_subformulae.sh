#!/bin/bash
#SBATCH --job-name=mistcf_subform
#SBATCH --partition=mit_preemptable,pi_ccoley
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/subform_%A_%a.out
#SBATCH --error=logs/subform_%A_%a.err
#SBATCH --array=0-17
#SBATCH --requeue

CHUNK_SIZE=1000
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))
MAX_TEST=17556
if [ $END_IDX -gt $MAX_TEST ]; then
    END_IDX=$MAX_TEST
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID, Task $SLURM_ARRAY_TASK_ID"
echo "Processing spectra $START_IDX to $END_IDX"
echo "=========================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5

cd /home/mrunali/MassSpecGym

SCRATCH=/home/mrunali/orcd/scratch/msg_benchmark
MSG=/home/mrunali/DiffMS/msg

python scripts/create_mistcf_subformulae.py \
    --predictions /orcd/pool/006/mrunali/mist-cf/results/mist_cf_msg/split_1_rnd1/preds_COMMON_optionb/top5_predictions.json \
    --spec-dir $MSG/spec_files \
    --labels $MSG/labels.tsv \
    --split $MSG/split.tsv \
    --output-base $SCRATCH/subformulae \
    --max-k 5 \
    --num-workers 16 \
    --start-idx $START_IDX \
    --end-idx $END_IDX

echo "Completed task $SLURM_ARRAY_TASK_ID"
