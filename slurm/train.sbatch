#!/bin/bash -l
# Train one config on a P100. Submit from the repo root:
#
#   sbatch slurm/train.sbatch configs/baseline.yaml
#   sbatch slurm/train.sbatch configs/baseline.yaml --set batch=32 imgsz=768
#   sbatch --job-name=n640 --time=12:00:00 slurm/train.sbatch configs/baseline.yaml
#
# If runs/detect/<name>/weights/last.pt already exists (job hit its time limit or crashed),
# re-submitting the same command RESUMES that run instead of failing.
#
#SBATCH --job-name=pneu-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1          # must say p100: a plain gpu:1 can land on the unusable K80 node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10         # half of gpu001's 20 cores, so both P100s can be used at once
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${1:?usage: sbatch slurm/train.sbatch <config.yaml> [--set key=value ...]}"
shift

module purge
module load python/3.11.5
source .venv/bin/activate
export SSL_CERT_FILE="$(python -m certifi)"   # cluster Python cannot find system CA certs; use certifi's bundle
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

NAME=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CONFIG")
RUN_DIR="runs/detect/$NAME"

if [ -f "$RUN_DIR/weights/last.pt" ]; then
    echo "Resuming $RUN_DIR"
    python src/train.py --resume "$RUN_DIR"
else
    python src/train.py --config "$CONFIG" --set workers=$((SLURM_CPUS_PER_TASK - 2)) "$@"
fi
