# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tunes Ultralytics YOLO26 as a single-class object detector (`pneumonia`) on the Kaggle RSNA Pneumonia Detection Challenge dataset. It outputs bounding boxes for pneumonia regions on chest X-rays. The end goals are a well-evaluated model, a results write-up with metrics charts for a portfolio site, and an inference website (planned: ONNX export run in the browser, living in `web/`).

`PIPELINE.md` explains each pipeline step, why it's done that way, and the current project status. Keep its status table updated as steps complete. `PROJECT_REVIEW.md` holds the original critique and a phased plan. Read it before changing the pipeline. Its key point: the existing `runs/detect/medical_imaging_v4` weights are a nano model at 640 px, trained in February 2026 on an older split, not the yolo26m/1024 config the notebook describes. Treat all existing metrics as untrustworthy until the model is retrained on a frozen train/val/test split.

## Keeping the docs current

After finishing any piece of work, update the relevant docs as part of the same step, without waiting to be asked:

| File | Holds | Update when |
|---|---|---|
| `PIPELINE.md` | How each pipeline step works and why; the **status table** and **What's next** | Any step completes, a decision is made, the pipeline or a config changes, results come in |
| `CLAUDE.md` | Commands, layout, environment, gotchas | Commands, scripts, layout or environment change; a new gotcha turns up |
| `Magnolia_HPC.md` | Cluster facts, survey output, implications | New cluster info (access, scratch, GPUs, admin replies) |
| `PROJECT_REVIEW.md` | The original critique and phased plan (historical) | Only to add a dated status note; don't rewrite the critique |
| `README.md` | Public summary, setup, results | Setup steps change; final results and demo are ready |
| `NOTES.md` | User's research notes (verbatim) | Only if the user asks |

Work in small steps and let the user make the decisions between them.

## Commands

Run from the repo root with `.venv\Scripts\python.exe`:

```
python src/make_splits.py                               # ONCE: frozen train/val/test split -> splits/*.csv (refuses to overwrite)
python src/prepare_data.py [--clean]                    # splits -> data/yolo (PNG + labels + data.yaml)
python src/train.py --config configs/smoke.yaml --no-wandb   # ~2-minute end-to-end pipeline check
python src/train.py --config configs/baseline.yaml      # real run -> runs/detect/<name>/
python src/train.py --config configs/baseline.yaml --set workers=8 batch=32   # per-machine overrides of `train:` keys
python src/train.py --resume runs/detect/<name>         # continue an interrupted run (same W&B run)
python src/evaluate.py runs/detect/<name>               # val: box mAP + image-level AUC/sens/spec + consistency checks
python src/evaluate.py runs/detect/<name> --split test --confirm-test   # final, one-time test evaluation
```

- The config's `name` is the run directory and the W&B run name. `train.py` refuses to reuse an existing name. A new experiment means a new config file.
- Everything under `train:` in a config goes verbatim to Ultralytics `model.train()`. `train.py` starts the W&B run itself (logging the config, split fingerprint and git commit), and Ultralytics' callback logs into that same run.
- Each run directory gets `config.yaml`, `provenance.json` (split fingerprint) and `wandb_id.txt`. `evaluate.py` refuses to evaluate a run trained on a different split.
- The split is 70/15/15 (18678/4003/4003 images, 22.5% positive), stratified on Target + view, seed 42. Never regenerate it with `--force` unless you are intentionally invalidating every trained model.

### Cluster (Magnolia HPC, SLURM): see `Magnolia_HPC.md`

```
DATA_DIR=$HOME/pneumonia-data bash slurm/setup_env.sh   # once, on the login node: py3.11 venv, torch 2.5.1+cu124, data/ symlink, base weights
sbatch slurm/prepare_data.sbatch                        # CPU job: Kaggle download + unzip + prepare_data.py
sbatch slurm/train.sbatch configs/<x>.yaml [--set k=v]  # P100 job; re-submitting resumes if last.pt exists
```

- The cluster runs CentOS 7 (glibc 2.17), which caps torch at 2.5.1. The laptop uses torch 2.10, and Ultralytics is pinned to the same version on both.
- The cluster installs from `requirements-cluster.txt` with `--only-binary=:all:`. Many of the laptop's exact pins (numpy 2.3+, pandas 3) have no glibc-2.17 wheels, and building from source fails. The resolved versions are frozen into `requirements-cluster.lock.txt`. Keep `ultralytics` and `albumentationsx` pinned identically in both requirements files.
- Always request `--gres=gpu:p100:1`. The K80 node is unusable.
- The cluster's Python (`module python/3.11.5`) can't find the system CA certificates, so urllib HTTPS fails with `CERTIFICATE_VERIFY_FAILED`. Every slurm script exports `SSL_CERT_FILE="$(python -m certifi)"` after activating the venv. Keep that line in any new cluster script.
- Data on the cluster currently lives in `$HOME/pneumonia-data` (`data/` symlink), because there's no scratch directory yet.
- `.gitignore` uses `/data` (no trailing slash) on purpose. On the cluster `data` is a symlink, i.e. a file, and `data/` only matches directories.
- A `.env` copied from Windows has CRLF line endings. Sourcing it in bash fails (`$'\r': command not found`, shown as just `: command not found`) and leaves `\r` on the values. `prepare_data.sbatch` loads it through `tr -d '\r'`. Do the same anywhere else `.env` is sourced, or fix the file once with `sed -i 's/\r$//' .env`.
- `sacct` may fail with "Connection refused" (the cluster's accounting DB is down). Read the job log in `slurm/logs/` instead.
- `requirements-cluster.lock.txt` is generated on the cluster by `setup_env.sh` and **should** be committed from there.
- Compute nodes have internet, so W&B logs online.
- Submit from the repo root. Logs go to `slurm/logs/`.
- In an interactive login shell, activate first: `module load python/3.11.5 && source .venv/bin/activate`. Otherwise `wandb`, `yolo` and the project's `python` aren't on PATH. Job scripts do this themselves.

## Layout

- `notebooks/archive/main.ipynb` is the original all-in-one notebook (download → EDA → preprocessing → augmentation → training → eval). It is kept for reference. Its paths are rewritten relative to `notebooks/archive/` (`../../data/...`), and its cells depend on in-memory state from earlier cells.
- `NOTES.md` holds the research notes and TODOs copied from the end of that notebook.
- `src/` holds the pipeline scripts, with shared paths in `src/common.py`. `configs/` holds one YAML per experiment. `splits/` holds the committed patientId CSVs with view/age/sex metadata.
- Still planned: `src/export.py` (ONNX), `web/` (the inference site, an empty placeholder for now), and a results-analysis notebook (charts for the site). EDA is complete in the archived notebook and is not being redone.
- `assets/` holds images for the README and site.
- Gitignored and kept local only: `data/raw/` (extracted Kaggle data), `data/yolo/` (generated YOLO dataset), `models/` (base weights such as `yolo26m.pt`), `runs/` (Ultralytics outputs), `wandb/`, `.venv/`, `.env`.

## Environment

- Python 3.13 venv at `.venv/`. Run scripts with `.venv\Scripts\python.exe`.
- Setup: install torch first (`pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130`), then `pip install -r requirements.txt`. All versions are pinned.
- GPU: an RTX 3050 Laptop with 4 GB VRAM. Large models at imgsz 1024 (for example yolo26m with batch 8) will probably run out of memory. Prefer n/s models at 512–640, or use `batch=-1` (auto batch).
- `albumentationsx` (in requirements) is imported as `albumentations`.
- `.env` provides `KAGGLE_USERNAME` / `KAGGLE_KEY`. W&B is used for experiment tracking. Prefer Ultralytics' built-in W&B logging over manual `wandb.log` of hyperparameters, so the logged config can't drift from the real training args.

## Data pipeline facts

- Raw: `data/raw/stage_2_train_images/{patientId}.dcm`, plus `stage_2_train_labels.csv` (pixel `x, y, width, height`, top-left origin, one row per box, so one patient can have several rows). Every image is 1024×1024.
- YOLO dataset `data/yolo/` (built by `prepare_data.py`): `images/{train,val,test}/*.png` (lossless, full 1024 resolution), `labels/{train,val,test}/*.txt`, and `data.yaml` (`nc: 1`, `names: [pneumonia]`). Negative images get an **empty** `.txt` label file. There is no offline augmentation; Ultralytics augments at train time.
- `prepare_data.py` skips files that already exist. After changing preprocessing, rebuild with `--clean`.

## Gotchas

- The Ultralytics confusion matrix is indexed `[predicted, true]` and has an extra background row/column.
- RSNA boxes go straight into `matplotlib.patches.Rectangle((x, y), w, h)` over `imshow` with no transform.
- mAP50 is not the RSNA competition metric (mean precision over IoU 0.4–0.75 per image, where any prediction on a negative image scores 0).
