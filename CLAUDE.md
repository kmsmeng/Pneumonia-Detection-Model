# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tunes Ultralytics YOLO26 as a single-class object detector (`pneumonia`) on the Kaggle RSNA Pneumonia Detection Challenge dataset. It outputs bounding boxes for pneumonia regions on chest X-rays. The end goals are a well-evaluated model, a results write-up with metrics charts for a portfolio site, and an inference website (planned: ONNX export run in the browser, living in `web/`).

`PROJECT_REVIEW.md` holds a critique of the current state and a phased plan. Read it before changing the pipeline. Its key point: the existing `runs/detect/medical_imaging_v4` weights are a nano model at 640 px, trained in February 2026 on an older split, not the yolo26m/1024 config the notebook describes. Treat all existing metrics as untrustworthy until the model is retrained on a frozen train/val/test split.

## Layout

- `notebooks/archive/main.ipynb` is the original all-in-one notebook (download → EDA → preprocessing → augmentation → training → eval). It is kept for reference. Its paths are rewritten relative to `notebooks/archive/` (`../../data/...`), and its cells depend on in-memory state from earlier cells.
- `NOTES.md` holds the research notes and TODOs copied from the end of that notebook.
- Planned (not yet created): `src/` scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `export.py`) driven by `configs/*.yaml`, `splits/` (committed patientId CSVs per split), `notebooks/01_eda.ipynb`, and `web/`.
- `assets/` holds images for the README and site.
- Gitignored and kept local only: `data/raw/` (extracted Kaggle data), `data/yolo/` (generated YOLO dataset), `models/` (base weights such as `yolo26m.pt`), `runs/` (Ultralytics outputs), `wandb/`, `myenv/` (the virtualenv), `.env`.

## Environment

- `myenv\Scripts\activate`, then `pip install -r requirements.txt`. Install PyTorch separately with the CUDA-specific command from pytorch.org (check the CUDA version with `nvidia-smi`).
- `albumentationsx` (in requirements) is imported as `albumentations`.
- `.env` provides `KAGGLE_USERNAME` / `KAGGLE_KEY`. W&B is used for experiment tracking. Prefer Ultralytics' built-in W&B logging over manual `wandb.log` of hyperparameters, so the logged config can't drift from the real training args.

## Data pipeline facts

- Raw: `data/raw/stage_2_train_images/{patientId}.dcm`, plus `stage_2_train_labels.csv` (pixel `x, y, width, height`, top-left origin, one row per box, so one patient can have several rows). Every image is 1024×1024.
- YOLO dataset `data/yolo/`: `images/{train,val}`, `labels/{train,val}`, and `data.yaml` (`nc: 1`, `names: [pneumonia]`, paths relative to the yaml). Negative images get an **empty** `.txt` label file. Offline-augmented positives are named `{patientId}(1|2|3).jpg` (train only).
- Generation steps skip files that already exist. After changing preprocessing, delete the affected outputs and `data/yolo/labels/*.cache`.
- The current split is 80/20, stratified on `Target + ViewPosition`, `random_state=42`.

## Gotchas

- The Ultralytics confusion matrix is indexed `[predicted, true]` and has an extra background row/column.
- RSNA boxes go straight into `matplotlib.patches.Rectangle((x, y), w, h)` over `imshow` with no transform.
- mAP50 is not the RSNA competition metric (mean precision over IoU 0.4–0.75 per image, where any prediction on a negative image scores 0).
