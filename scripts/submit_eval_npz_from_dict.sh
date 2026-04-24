#!/bin/bash
#SBATCH --job-name=npz_from_dict_eval
#SBATCH --partition=pi_ccoley,mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/npz_from_dict_eval_%j.out
#SBATCH --error=logs/npz_from_dict_eval_%j.err

# Evaluate the pre-made binary fps in msg_mist_test_thr0.187_fp4096.npz using
# FromDictRetrieval + trainer.test. This reuses base.py's evaluate_retrieval_step
# (retrieval_hit_rate), so hit rates are computed identically to job 12427444.
#
# Similarity: tanimoto (npz fps are binary; cosine would misrank).
# Candidates: bonus (formula) to match job 12427444 setup.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate msg_v1.5
cd /home/mrunali/MassSpecGym
mkdir -p logs results/mist_fps

NPZ="results/mist_fps/msg_mist_test_thr0.187_fp4096.npz"
DCT_PATH="results/mist_fps/msg_mist_test_thr0.187_fp4096.pkl"
SPLIT_FILTERED="results/mist_fps/split_npz_present.tsv"

if [ ! -f "$DCT_PATH" ]; then
    echo "Converting $NPZ -> $DCT_PATH"
    python scripts/npz_to_dict.py --npz "$NPZ" --out "$DCT_PATH"
else
    echo "Reusing existing $DCT_PATH"
fi

# npz covers 17082 of 17556 test specs; filter split.tsv to only the present ones
# so RetrievalDataset never asks FromDictRetrieval for a missing identifier.
python - <<PY
import numpy as np, pandas as pd
npz = np.load("$NPZ", allow_pickle=True)
present = set(map(str, npz["spec_id"]))
split = pd.read_csv("data/msg/split.tsv", sep="\t")
keep = (split["split"] != "test") | (split["name"].astype(str).isin(present))
split.loc[keep].to_csv("$SPLIT_FILTERED", sep="\t", index=False)
n_test_before = int((split["split"] == "test").sum())
n_test_after = int(((split["split"] == "test") & split["name"].astype(str).isin(present)).sum())
print(f"split: test kept {n_test_after}/{n_test_before} (dropped {n_test_before-n_test_after} with no fp)")
PY

python scripts/run.py \
    --job_key  "from_dict_npz_thr0.187" \
    --run_name "from_dict_npz_thr0.187" \
    --no_wandb \
    --test_only \
    --skip_mces_test \
    --task retrieval \
    --model from_dict \
    --dct_path "$DCT_PATH" \
    --fp_similarity tanimoto \
    --candidates_pth bonus \
    --split_pth "$SPLIT_FILTERED" \
    --fp_size 4096 \
    --accelerator gpu \
    --devices 1 \
    --batch_size 32 \
    --num_workers 8 && echo "Done: from_dict npz eval"
