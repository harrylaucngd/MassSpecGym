#!/bin/bash
#SBATCH --job-name=mist_fps_eval
#SBATCH --partition=pi_ccoley,ou_cheme,ou_cheme_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mist_fps_eval_%j.out
#SBATCH --error=logs/mist_fps_eval_%j.err

# Usage: sbatch --dependency=afterok:<predict_job_id> scripts/submit_mist_fps_eval.sh
echo "Job: $SLURM_JOB_ID"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym
mkdir -p results/mist_fps_eval

# Download candidate JSONs via HuggingFace hub (cached after first run)
MASS_CANDS=$(python3 -c "from massspecgym.utils import hugging_face_download; print(hugging_face_download('molecules/MassSpecGym_retrieval_candidates_mass.json'))")
BONUS_CANDS=$(python3 -c "from massspecgym.utils import hugging_face_download; print(hugging_face_download('molecules/MassSpecGym_retrieval_candidates_formula.json'))")
echo "Mass candidates: $MASS_CANDS"
echo "Bonus candidates: $BONUS_CANDS"

echo "--- Eval: mass candidates (GT + MISTCF rank1) ---"
python scripts/eval_retrieval_from_fps.py \
    --fp-dir results/mist_fps \
    --labels data/msg/labels.tsv \
    --split data/msg/split.tsv \
    --candidates-json "$MASS_CANDS" \
    --output-dir results/mist_fps_eval/mass \
    --at-ks 1 5 10 20

echo "--- Eval: bonus formula candidates (GT + MISTCF rank1) ---"
python scripts/eval_retrieval_from_fps.py \
    --fp-dir results/mist_fps \
    --labels data/msg/labels.tsv \
    --split data/msg/split.tsv \
    --candidates-json "$BONUS_CANDS" \
    --output-dir results/mist_fps_eval/bonus \
    --at-ks 1 5 10 20

echo "Done"
