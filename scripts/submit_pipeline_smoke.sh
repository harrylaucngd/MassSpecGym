#!/bin/bash
#SBATCH --job-name=msg_smoke
#SBATCH --partition=pi_ccoley,ou_cheme,ou_cheme_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/pipeline_smoke_%j.out
#SBATCH --error=logs/pipeline_smoke_%j.err

# Smoke-tests the run.py inferred_formula code path using 5-spectrum debug dataset.
echo "Job: $SLURM_JOB_ID"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym
mkdir -p logs data/test_results/retrieval

echo "--- Test 1: mass baseline (random, debug) ---"
python scripts/run.py \
    --job_key smoke_mass_$$ \
    --run_name smoke_random_mass \
    --task retrieval \
    --model random \
    --devices 1 \
    --accelerator cpu \
    --test_only \
    --no_wandb \
    --num_workers 0 \
    --debug

echo "--- Test 2: inferred_formula pipeline (random, debug) ---"
python scripts/run.py \
    --job_key smoke_inferred_$$ \
    --run_name smoke_random \
    --inferred_formula \
    --inferred_formula_pth data/debug/example_5_spectra_inferred_formula.json \
    --task retrieval \
    --model random \
    --devices 1 \
    --accelerator cpu \
    --test_only \
    --no_wandb \
    --num_workers 0 \
    --debug

echo "--- PKL files generated ---"
ls -lh data/test_results/retrieval/smoke_*.pkl 2>/dev/null || echo "No pkls found"
echo "Done"
