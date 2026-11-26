#!/bin/bash -l
# Download (if needed) + unzip the RSNA data, then build data/yolo from the committed splits.
# CPU-only job. Submit from the repo root:
#
#   sbatch slurm/prepare_data.sbatch
#
#SBATCH --job-name=pneu-prep
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.11.5
source .venv/bin/activate
export SSL_CERT_FILE="$(python -m certifi)"   # cluster Python cannot find system CA certs; use certifi's bundle

ZIP=data/rsna-pneumonia-detection-challenge.zip
if [ ! -d data/raw/stage_2_train_images ]; then
    if [ ! -f "$ZIP" ]; then
        set -a; [ -f .env ] && source .env; set +a   # KAGGLE_USERNAME / KAGGLE_KEY
        kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/
    fi
    echo "Unzipping $ZIP ..."
    mkdir -p data/raw
    python -m zipfile -e "$ZIP" data/raw
fi

python src/prepare_data.py --workers "$SLURM_CPUS_PER_TASK"
