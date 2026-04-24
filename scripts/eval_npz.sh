#!/bin/bash
#SBATCH --job-name=mist_npz_eval
#SBATCH --partition=pi_ccoley,ou_cheme,ou_cheme_preemptable
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mist_npz_eval_%j.out
#SBATCH --error=logs/mist_npz_eval_%j.err

echo "Job: $SLURM_JOB_ID"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym
mkdir -p logs results/mist_fps_eval/bonus_colleague

# Convert NPZ → PT (spec_id -> float32 tensor)
python3 - << 'PYEOF'
import numpy as np, torch
npz = np.load('results/mist_fps/msg_mist_test_thr0.187_fp4096.npz', allow_pickle=True)
fps = {str(s): torch.tensor(f.astype('float32'))
       for s, f in zip(npz['spec_id'], npz['fingerprint'])}
torch.save(fps, 'results/mist_fps/fingerprints_colleague.pt')
print(f'Saved {len(fps)} fingerprints → fingerprints_colleague.pt')
PYEOF

# Get cached formula candidates path
FORMULA_CANDS=$(python3 -c "
from massspecgym.utils import hugging_face_download
print(hugging_face_download('molecules/MassSpecGym_retrieval_candidates_formula.json'))
")
echo "Formula candidates: $FORMULA_CANDS"

# Evaluate colleague fingerprints only (symlink into a temp dir)
mkdir -p results/mist_fps_colleague_only
ln -sf /home/mrunali/MassSpecGym/results/mist_fps/fingerprints_colleague.pt \
       results/mist_fps_colleague_only/fingerprints_colleague.pt

python scripts/eval_retrieval_from_fps.py \
    --fp-dir results/mist_fps_colleague_only \
    --labels data/msg/labels.tsv \
    --split data/msg/split.tsv \
    --candidates-json "$FORMULA_CANDS" \
    --output-dir results/mist_fps_eval/bonus_colleague \
    --at-ks 1 5 10 20

echo "Done: $SLURM_JOB_ID"
