#!/bin/bash
#SBATCH --job-name=tar_subform
#SBATCH --partition=ou_cheme_preemptable,ou_cheme,mit_preemptable,pi_ccoley
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tar_subform_%j.out
#SBATCH --error=logs/tar_subform_%j.err

set -euo pipefail
cd /home/mrunali/MassSpecGym/data/msg/subformulae

OUT=/home/mrunali/orcd/scratch/default_subformulae.tar.gz
mkdir -p "$(dirname "$OUT")"

echo "Starting: $(date)"
echo "Source: $(pwd)/default_subformulae"
echo "Output: $OUT"
du -sh default_subformulae

tar -cf - default_subformulae | pigz -p 8 > "$OUT"

echo "Done: $(date)"
ls -la "$OUT"
# SHA256 so the collaborator can verify the transfer
sha256sum "$OUT"
