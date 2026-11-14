#!/bin/bash
# One-time setup on Magnolia. Run on the LOGIN node, from the repo root:
#
#   bash slurm/setup_env.sh
#
# - Python 3.11 venv in .venv/ (code + venv live in $HOME)
# - torch 2.5.1 (cu124) + requirements-cluster.txt, pre-built wheels only (CentOS 7 / glibc 2.17);
#   CUDA 12.4 matches the P100 node's driver (550)
# - data/ -> symlink to /scratch (home is 99% full; the dataset needs ~15 GB)
# - downloads the base YOLO weights into models/
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/scratch/$USER/pneumonia-data}"

module purge
module load python/3.11.5
python3 --version

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# One resolve for everything. --only-binary=:all: makes pip pick versions with pre-built wheels for
# CentOS 7 (glibc 2.17) instead of compiling from source (which fails here, e.g. numpy 2.4).
# PyPI is added as an extra index so torch's dependencies (numpy, ...) resolve from it.
pip install --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 torchvision==0.20.1 -r requirements-cluster.txt
pip check
pip freeze > requirements-cluster.lock.txt   # exact versions on the cluster; commit this file
echo "Wrote requirements-cluster.lock.txt"

# Data on scratch, linked into the repo so the scripts' data/ paths keep working.
mkdir -p "$DATA_DIR"
if [ -e data ] && [ ! -L data ]; then
    echo "ERROR: data/ exists and is not a symlink. Move its contents to $DATA_DIR and remove it first." >&2
    exit 1
fi
ln -sfn "$DATA_DIR" data
echo "data -> $(readlink data)"

# Base weights (models/ is gitignored, so they aren't in the clone).
mkdir -p models slurm/logs
(cd models && python -c "from ultralytics import YOLO; [YOLO(w) for w in ('yolo26n.pt', 'yolo26s.pt')]")

python -c "import torch, ultralytics, cv2, pandas, pydicom; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__)"
echo
echo "Setup done. Next:"
echo "  1. wandb login                      (once; stores your API key)"
echo "  2. copy your .env (Kaggle keys) into the repo root"
echo "  3. sbatch slurm/prepare_data.sbatch"
