# Pipeline and Project Status

How the project works end to end: what each step does, why it's done that way, and where things stand.

- Cluster environment details: [`Magnolia_HPC.md`](Magnolia_HPC.md)
- Original critique and phased plan: [`PROJECT_REVIEW.md`](PROJECT_REVIEW.md)

*Last updated: 2026-11-25*

---

## The big picture

```
Kaggle DICOMs ──► 1. make_splits ──► 2. prepare_data ──► 3. train ──► 4. evaluate
 (raw X-rays)      (who goes where)    (convert to YOLO)    (fit model)   (score it honestly)
```

| Step | Script | Runs | Output |
|---|---|---|---|
| 1. Split | `src/make_splits.py` | **Once, ever** | `splits/{train,val,test}.csv` (committed to git) |
| 2. Prepare | `src/prepare_data.py` | Once per machine (laptop, cluster) | `data/yolo/` (PNG images, labels, `data.yaml`) |
| 3. Train | `src/train.py` + `configs/*.yaml` | Once per experiment | `runs/detect/<name>/` |
| 4. Evaluate | `src/evaluate.py` | After each training run | `runs/detect/<name>/eval_<split>.json` |

### Where EDA fits

EDA is **not** part of this pipeline. It was a one-time study, already done in `notebooks/archive/main.ipynb` (class balance, the 3-class detail labels, age/sex/view distributions, bbox sizes and the location heatmap). Its findings shaped decisions below, for example *"stratify on view"* and *"AP vs PA matters"*. It isn't being redone. A separate results-analysis notebook will come later to make charts for the website.

---

## Step 1: Splitting the data (`src/make_splits.py`)

**What it does**

1. Reads `stage_2_train_labels.csv` and collapses it to **one row per patient (X-ray image)**, with `Target` (0/1) and `n_boxes`. A patient with pneumonia can have several boxes, and therefore several rows in the raw CSV.
2. Merges in the 3-class detail label (`Normal`, `Lung Opacity`, `No Lung Opacity / Not Normal`).
3. Reads every DICOM **header** (not the pixels, so it's fast) to get `view` (AP/PA), `age`, `sex`, `rows` and `cols`, and saves them alongside the split. That lets us break results down by view, age or sex later without touching the DICOMs again.
4. Splits **70 / 15 / 15**, **stratified on `Target + view`**, `random_state=42`: first test is carved off, then val from the rest.
5. Asserts that **no patient appears in two splits**.

**Result**

| Split | Images | Pneumonia (Target=1) | % positive | GT boxes | % AP view |
|---|---|---|---|---|---|
| train | 18,678 | 4,208 | 22.5% | 6,680 | 45.6% |
| val | 4,003 | 902 | 22.5% | 1,424 | 45.6% |
| test | 4,003 | 902 | 22.5% | 1,451 | 45.6% |

**Why it's done this way**

- **Per patient:** the same patient in train and val would leak information and inflate the scores.
- **Stratified on Target + view:** each split gets the same disease mix *and* the same mix of AP/PA X-rays. AP views come from sicker, often bedridden patients and look different, so an uneven mix would skew the results.
- **Three sets, each with a role:**
  - **train:** what the model learns from.
  - **val:** for comparing experiments and picking the best checkpoint. Used repeatedly.
  - **test:** **locked until the very end.** Evaluated once, for the final reported numbers. Using it to make decisions would make it optimistic, just like val.
- **Frozen and committed to git:** every model is trained and judged on exactly the same split, so experiments are comparable. The old project's results were invalid partly because the split changed between training and evaluation. The script refuses to overwrite the split unless you pass `--force`.

---

## Step 2: Preparing the data (`src/prepare_data.py`)

**What it does**

1. **DICOM → PNG.** Each X-ray's pixels are saved as an 8-bit grayscale PNG, **lossless** and at **full 1024×1024** resolution. The old pipeline used JPEG (quality 75), which throws away fine texture, and texture is exactly the signal for lung opacities. Full resolution keeps the option of training at 768 or 1024 px later.
2. **Boxes → YOLO format.** RSNA gives `x, y, width, height` in pixels with a top-left origin. YOLO wants `class x_center y_center width height`, normalized to 0–1. Boxes are clipped to the image edges, and images with several boxes keep all of them (one line each).
3. **Healthy images get an empty `.txt` label file.** That tells YOLO "nothing to find in this image". These images teach the model what *not* to flag.
4. **Writes `data/yolo/data.yaml`**, which tells YOLO where train/val/test are, plus `nc: 1` and `names: [pneumonia]`.
5. **Checks itself:** the number of images and of positive (non-empty) label files must match the split CSVs exactly, or it stops. It also deletes stale Ultralytics `*.cache` files.

**Layout produced** (~11 GB):

```
data/yolo/
├── data.yaml
├── images/{train,val,test}/<patientId>.png
└── labels/{train,val,test}/<patientId>.txt     (empty for healthy images)
```

It's safe to re-run: files that already exist are skipped. `--clean` deletes and rebuilds everything.

### What it deliberately does NOT do anymore

**No offline augmentation.** The old pipeline saved 3 edited copies of every pneumonia image to disk. Now augmentation happens **on the fly during training**. Each epoch, Ultralytics randomly:

| Augmentation | Setting | Why |
|---|---|---|
| Horizontal flip | `fliplr: 0.5` | Lungs are roughly symmetric |
| Shift / scale | `translate: 0.1`, `scale: 0.5` | Simulates positioning and patient size |
| Brightness | `hsv_v: 0.4` | Different X-ray machines and exposures |
| Mosaic (4 images tiled) | `mosaic: 1.0`, off for the last 10 epochs | More context variety, good for small objects |
| Hue / saturation | **off** (`hsv_h: 0`, `hsv_s: 0`) | X-rays are grayscale, so colour jitter is meaningless |
| Vertical flip, rotation | **off** | Upside-down lungs aren't realistic, and rotation inflates boxes |

So the model sees slightly different versions of each image every epoch, without extra files on disk.

**No class-imbalance fix (yet).** The baseline trains on the natural mix of **22.5% pneumonia / 77.5% healthy**. This is on purpose:

- In object detection, healthy images aren't wasted. They teach the model what "no pneumonia" looks like, which reduces false alarms.
- Oversampling pneumonia (the old 3× copies) may increase false positives on healthy scans, and nobody has measured whether it helped.

Handling the imbalance is therefore **an experiment, not an assumption**. After the baseline, we train the same setup with oversampling or fewer negatives and compare (see [What's next](#whats-next)).

---

## Step 3: Training (`src/train.py` + `configs/`)

**Configs are the single source of truth.** Each experiment is one YAML file in `configs/`. Everything under `train:` is passed verbatim to Ultralytics `model.train()`.

| Config | Purpose |
|---|---|
| `configs/smoke.yaml` | Pipeline check: 2% of train, 1 epoch, about 2 minutes. Its metrics are meaningless. |
| `configs/baseline.yaml` | First real model: `yolo26n`, 640 px, 30 epochs, early stopping (patience 10), the augmentation above |

**What `train.py` records for every run** (in `runs/detect/<name>/`):

- `config.yaml`: the exact config used, **including** any `--set` overrides
- `provenance.json`: the split fingerprint (a hash of `splits/*.csv`), the git commit, the config file and the overrides
- `wandb_id.txt`: the W&B run ID, so a resumed run and the later evaluation land in the **same** W&B run
- Ultralytics output: `weights/best.pt`, `weights/last.pt`, `results.csv`, curves, sample batches

**Rules it enforces:**

- A run name can't be reused, so a new experiment needs a new config `name`. That way old results are never overwritten.
- `--set key=value` overrides per machine (e.g. `--set workers=8 batch=32`). The override is recorded, so the record always matches what actually ran.
- `--resume runs/detect/<name>` continues an interrupted run (same settings, same W&B run).
- W&B logging: `train.py` starts the W&B run itself (with the config, split fingerprint and git commit), and Ultralytics' built-in logger writes losses, metrics and plots into it. The logged config can't drift from the real one, which was a problem in the old notebook.

---

## Step 4: Evaluating (`src/evaluate.py`)

Run on **val** after every experiment. Run on **test** only once, at the very end (`--split test --confirm-test`).

**What it reports:**

| Level | Metrics | Question it answers |
|---|---|---|
| **Box** | mAP50, mAP50-95, precision, recall | Are the boxes in the right places? |
| **Box** | Confusion matrix (conf 0.25, IoU 0.45): TP / FP / FN | How many boxes were found, false, or missed? |
| **Image** | ROC-AUC, PR-AUC | Overall, how well does it separate pneumonia from healthy X-rays? |
| **Image** | Sensitivity, specificity at a threshold (default 0.25) | Does it **catch** pneumonia patients? Does it avoid **false alarms** on healthy ones? |
| **Image, by view** | The same, split into AP and PA | Does it work equally on both X-ray types? |

The image-level score is the **highest box confidence in the image** (0 if it found no box). This is the clinically meaningful question: *"does this patient have pneumonia?"*

**Safety checks** (these catch the kinds of bugs that invalidated the old results):

- It refuses to evaluate a run that was trained on a **different split** (via the split fingerprint).
- It warns if the confusion matrix doesn't account for every ground-truth box, or if mAP > 0 while TP = 0 (the old "impossible" matrix).
- It won't touch the test set without `--confirm-test`.

Results are saved to `runs/detect/<name>/eval_<split>.json` and added to the run's W&B summary.

> Note: mAP50 is **not** the RSNA competition metric (mean precision over IoU 0.4–0.75 per image, where any
> prediction on a healthy image scores 0). That metric is planned for later.

---

## Where training runs: laptop vs. cluster

| | Laptop | Magnolia HPC cluster |
|---|---|---|
| GPU | RTX 3050, 4 GB | Tesla P100, 12 GB (2 available, shared) |
| Used for | Smoke tests, evaluation, the website | Real training runs |
| OS / torch | Windows / torch 2.10 (CUDA 13.0) | CentOS 7 / torch **2.5.1** (CUDA 12.4), the newest torch that runs on CentOS 7 |
| Packages | `requirements.txt` (exact pins) | `requirements-cluster.txt` (pre-built files only), exact versions in `requirements-cluster.lock.txt` |
| Data loading | 2 workers (more ran out of memory) | 8 workers (10 CPU cores per job) |

`ultralytics` and `albumentationsx` are pinned to the **same version** on both, so training behaves identically. A model trained on the cluster loads fine on the laptop.

### Expected training time (baseline: yolo26n, 640 px, 30 epochs)

| | Per epoch | 30 epochs | Basis |
|---|---|---|---|
| Laptop (RTX 3050, 2 workers) | ~20 min | ~10 h | Measured: ~1 s per batch of 16 in the smoke run and earlier runs |
| Cluster (1× P100, 8 workers) | ~6–12 min | ~3–6 h | **Estimate**; replace with the measured `s/it` from the cluster smoke run |

- The laptop is likely bottlenecked on data loading (only 2 workers fit in RAM), not on the GPU. The cluster's 8 workers keep the GPU busy. GPU vs. GPU, a P100 and an RTX 3050 Laptop are roughly comparable.
- The cluster's real advantages are 12 GB VRAM (bigger models and images), **two P100s for parallel experiments**, long uninterrupted runs, and a free laptop.
- A bigger config (e.g. yolo26s at 640 px, ~3× the compute) is roughly 10–18 h on a P100, which is impractical on the laptop.
- Shared GPUs mean a queue wait (check with `squeue -p gpu`). If reading images from the home folder is slow, try `--set cache=ram`.

### Cluster scripts (`slurm/`)

| Script | Where | What it does |
|---|---|---|
| `slurm/setup_env.sh` | Login node, once | Python 3.11 venv, torch 2.5.1 + packages, `data/` link to the data location, downloads base weights |
| `slurm/prepare_data.sbatch` | CPU job | Downloads the dataset from Kaggle (if needed), unzips it, runs `prepare_data.py` |
| `slurm/train.sbatch` | GPU job (1× P100, 10 CPUs, 60 GB, up to 2 days) | Runs `train.py` with a config. **Re-submitting resumes** the run if `last.pt` exists. |

Usage (from the repo root on Magnolia):

```bash
DATA_DIR=$HOME/pneumonia-data bash slurm/setup_env.sh   # once
wandb login                                             # once
sbatch slurm/prepare_data.sbatch                        # once
sbatch slurm/train.sbatch configs/smoke.yaml --no-wandb # pipeline check
sbatch slurm/train.sbatch configs/baseline.yaml         # real run
squeue -u $USER                                         # job status
tail -f slurm/logs/<job>.out                            # live log
```

### Where the data lives

- Code and the venv live in the home folder (`/homes/01/<username>`).
- `data/` is a **link** to wherever the dataset is stored. For now that's `$HOME/pneumonia-data`, because there's no personal scratch folder yet.
- Scratch (`/scratch`, fast, 11 TB free) would be better for heavy image reading. A scratch directory has been requested from the admin (2026-11-14). If it's granted, moving there is just: move the folder, then `ln -sfn <new path> data`.
- If reading from home turns out to be slow, `--set cache=ram` loads all training images into RAM (~23 GB at 640 px, which fits in the job's 60 GB).

---

## Project status

| | Status | Notes |
|---|---|---|
| EDA | ✅ Done | In the archived notebook; not redone |
| Frozen train/val/test split | ✅ Done | `splits/`, committed |
| Data pipeline (`prepare_data.py`) | ✅ Done | Built and verified on the laptop |
| Train / eval scripts | ✅ Done | Tested end to end on the laptop (smoke run) |
| Cluster job scripts | ✅ Written | `slurm/` |
| Cluster software install | ✅ Done | torch 2.5.1, every package, no conflicts |
| Cluster data setup | ⏳ Next | `DATA_DIR=$HOME/pneumonia-data bash slurm/setup_env.sh`, `wandb login`, copy `.env`, `sbatch slurm/prepare_data.sbatch` |
| Cluster smoke run | ⏳ | Checks that torch 2.5.1 + YOLO26 + P100 + paths work together |
| Baseline training + val evaluation | ⏳ | The first trustworthy numbers |
| Scratch directory / A100 access | ⏳ Waiting | Email sent to the cluster admin 2026-11-14; no reply as of 2026-11-25. Proceeding with the home folder + P100s. |

## What's next

In order, one step at a time:

1. **Cluster data setup**, then the **smoke run** on the cluster.
2. **Choose the baseline config.** Currently `yolo26n` at 640 px (sized for the laptop). The 12 GB P100 makes `yolo26s` at 640–768 px an option.
3. **Train the baseline and evaluate it on val:** the first numbers we can trust.
4. **Experiments, one change at a time, compared on val:**
   - Class imbalance: natural mix vs. oversampled pneumonia vs. fewer healthy images
   - Image size: 512 / 640 / 768 / 1024
   - Model size: n / s / m
   - Small extras: cosine LR schedule, CLAHE contrast preprocessing, test-time augmentation
5. **Match the real task:** pick the confidence threshold on val, implement the RSNA competition metric, and optionally add an image-level classifier to cut false alarms.
6. **Final:** evaluate the chosen model **once** on test, export it to ONNX, write up the results charts, and build the inference website.
