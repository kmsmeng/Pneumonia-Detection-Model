# Project Review: Pneumonia Detection Model

*Written 2026-11-13. Based on `main.ipynb`, `runs/detect/*/args.yaml` and `results.csv`, and the git history.*

---

## TL;DR

The EDA and data pipeline are careful and well commented. The weakest part of the project is **evaluation integrity**. The model that every evaluation cell loads (`runs/detect/medical_imaging_v4/weights/best.pt`) is not the model the notebook says it trained, and it was not trained on the data split it is now evaluated on. Until that is fixed, none of the reported numbers can be trusted, including the 0.258 mAP50, the confusion matrix and the W&B artifact metadata.

**What to do first:** retrain once on the current split, evaluate on a held-out test set nobody has touched, and record the real config. Then start experimenting.

---

## 1. Critical issues (fix before anything else)

### 1.1 The evaluated model doesn't match what the notebook says it is

| | What the notebook / W&B says | What `runs/detect/medical_imaging_v4/args.yaml` and the files show |
|---|---|---|
| Model | `yolo26m` (20.4M params) | ~6.2 MB weights: a **nano**-sized model (yolo26m weights are ~44 MB) |
| Image size | 1024 | **640** |
| Epochs | 50 | **20** (resumed run) |
| Batch / optimizer / lr | 8 / AdamW / 0.0005 | **16 / auto / 0.01** |
| Date trained | — | Weights dated **2026-02-11** |

- The yolo26m / 1024 / AdamW run in cell 100 has **no output folder** anywhere in the repo (no `runs/detect/yolo26m-50epochs-AdamW-lr0.0005/`, no `rsna-pneumonia-detection/`). Either it never finished or its output went somewhere else.
- The W&B artifact `pneumonia-detector-yolo26m` uploads the v4 nano weights but labels them `yolo26m`, `epochs: 50`, `imgsz: 1024`. **Your experiment tracking is recording false information.**
- `wandb.summary` hard-codes `total_val_samples: 8943`, but the val set is 5337. `run_config` also hard-codes sample counts instead of computing them.

### 1.2 Probable data leakage in the evaluation

The v4 weights are from February. The stratified split (`random_state=42`, stratified on `Target + view`) and the augmentation were added in late May and June (see the commits). The February model was almost certainly trained on a **different train/val split**, so part of today's `images/val` was likely in its training data. Any `model.val()` on the current val set gives an inflated, meaningless number.

### 1.3 No held-out test set

The val set is used for early stopping, for picking `best.pt`, for the threshold and curve analysis, *and* as the final reported score. That is an optimistic estimate. `stage_2_test_images` has no labels, so you need to carve your own test split (see §5).

### 1.4 The confusion matrix analysis is contradictory

The markdown in cell 115 (which reads like a pasted chatbot answer: *"Your matrix is…"*) concludes TP = 0, FN = 587, FP = 1928. But:

- A model with **TP = 0 cannot have mAP50 = 0.25 or recall = 0.30.** Those numbers are incompatible, so something is wrong in how the matrix was produced or read.
- In Ultralytics the matrix is indexed `[predicted, true]`: rows are predictions, columns are ground truth. Read that way, `[[0, 587], [1928, 0]]` gives **FP = 587** and **FN = 1928**. 1928 is about the total number of ground-truth boxes in val (1202 positive images × ~1.5 boxes), meaning *every box was missed*. That points to a pipeline bug in that particular `val()` call, not just a weak model. Likely suspects: the model/split mismatch above, a stale `labels/*.cache`, or val run on a different image size than the model was trained on.
- **Action:** after retraining, print `val_metrics.box.map50` and the matrix from the *same* call and check that they agree. Delete `YOLO_format_data/labels/*.cache` whenever labels are regenerated.

---

## 2. Methodology issues

### 2.1 Offline augmentation strategy
- **Imbalance matters less for detection than for classification.** Negative images supply "background" examples. Duplicating each positive 3× changes the prior the model learns, which tends to cause **more false positives on healthy scans**. On RSNA that is heavily penalised and clinically costly.
- **It doubles up with the built-in Ultralytics augmentation.** `fliplr: 0.5`, `mosaic: 1.0`, `scale: 0.5`, `translate: 0.1` are already on. Your offline `HorizontalFlip` copy is redundant with `fliplr`.
- **`hsv_s: 0.7` / `hsv_h: 0.015` do nothing useful** on grayscale X-rays. Set them to 0.
- **Rotating ±10° inflates boxes.** Albumentations recomputes an axis-aligned box around the rotated box, so every rotated label gets looser. That adds label noise in a task where IoU ≥ 0.4–0.75 is what counts.
- **The offline copies are near-duplicates.** `patientX(1..3)` plus the original are 4 versions of the same patient. They inflate the epoch count without adding much information, and they cost ~15k extra JPEGs on disk.
- **Test it, don't assume it.** Train the same config with and without offline augmentation and compare on the test set. It is quite possible "no offline aug, built-in aug only" wins.

### 2.2 Training duration and overfitting
- `medical_imaging_50_epo` peaked at **epoch 21** (mAP50 0.263). After that `train/box_loss` keeps falling (1.75 → 1.37) while `val/box_loss` rises (1.97 → 2.18). That is overfitting. The "train for 100+ epochs" item in the TODO list isn't supported by your own data. A bigger model, better aug or more data are more promising levers than more epochs.
- Every run shows a large train/val loss gap (val cls_loss around 3.0 vs train around 2.3).

### 2.3 Image preprocessing
- **DICOM → JPEG is lossy.** Fine opacity texture is the signal here. Use PNG (lossless), or at least `quality=95+`. `Image.save(..., 'JPEG')` defaults to quality 75.
- **The TODO "downsample to 224×224" would likely hurt.** Opacities are diffuse and low-contrast. RSNA solutions typically used 512–1024. Test 512 vs 768 vs 1024, not 224.
- **CLAHE / histogram equalisation is unexplored.** It is a common, cheap preprocessing step for chest X-rays.

### 2.4 Metrics don't match the real task
- **mAP50 is not the RSNA metric.** The competition scored mean precision over IoU thresholds 0.4–0.75, per image, and an image with no ground truth scores 0 if it gets *any* prediction. Your number is not comparable to the leaderboard. The winning stage-2 scores were around 0.25 *on that metric*, so even a real mAP50 of 0.25 says little about where you stand.
- **There are no image-level metrics.** The clinically useful question is "does this patient have pneumonia?". Compute image-level **sensitivity, specificity, ROC-AUC**, using the max box confidence per image as the score. That also shows how many healthy scans get false alarms.
- **The confidence threshold is never tuned.** Pick it from the F1 curve or the competition metric on val, then report the test result at that fixed threshold.

### 2.5 Inference demo uses training images
`run_inference` samples from the full `df[df.Target == 1]`, which is mostly *training* images, so the demo shows the model on data it has seen. Sample from the val/test patient IDs instead.

Also, the `transforms.ToTensor()` comment ("required to pass into the YOLO model") is wrong. Ultralytics accepts paths and numpy arrays and does its own preprocessing.

### 2.6 Unfinished analysis
- The "Post Evaluation Heatmap" cell (126) is a copy of the *label* heatmap. The predicted-box heatmap described in cell 147 was never written.
- The train-vs-val metrics comparison (cell 130) is empty.
- The `view` (AP/PA) imbalance was found in EDA and used for stratification, but performance was never **broken down by view, sex or age**. That is an easy, interesting fairness/robustness analysis. AP views come from sicker, often supine patients and behave differently.

---

## 3. Engineering and reproducibility

### 3.1 One huge stateful notebook
- `main.ipynb` has 160 cells and is 3.5 MB with outputs. It mixes download, EDA, preprocessing, training, evaluation and personal research notes.
- **Cells aren't safe to re-run.** For example, cell 60 does `df = pd.concat([df, dcm_metadata['view']], axis=1)`. Running it twice creates a duplicate `view` column and breaks `stratify_group`. Several cells mutate `df` in place.
- Paths like `data/pneumonia_dataset/stage_2_train_images/{Id}.dcm` and `runs/detect/medical_imaging_v4/...` are hard-coded and repeated many times.
- About 15 empty cells, plus dead and commented-out code (the ThreadPool square example, the COCO8 val snippet, the wine-reviews download).

**Suggested structure:**
```
src/
  config.py         # paths, split seed, hyperparams (single source of truth)
  prepare_data.py   # DICOM -> PNG, labels, split, data.yaml  (idempotent CLI)
  augment.py        # optional offline aug
  train.py          # reads config, trains, logs config to W&B from the SAME dict
  evaluate.py       # test-set eval, image-level metrics, RSNA metric
  predict.py        # inference + visualisation
notebooks/
  eda.ipynb         # (you've already started this split — good)
  analysis.ipynb    # results, error analysis, heatmaps
configs/
  yolo26n_640.yaml, yolo26m_1024.yaml ...
```
A script per stage means you can rerun one stage without carrying hidden notebook state along with it.

### 3.2 Config drift
`run_config` and the `model.train(...)` arguments are written out separately and have already drifted apart (§1.1). Build the train kwargs **from** the config dict (`model.train(**cfg['train'])`) so they can't diverge. Ultralytics' W&B integration also logs the real args.

### 3.3 W&B usage
- `wandb.init(name=..., resume='allow')`: resuming matches on run **id**, not name. Without a fixed `id=`, every notebook session starts a new run, which is why `wandb/` holds 8 runs.
- `project=WANDB_PROJECT` in `model.train()` makes Ultralytics write to a folder named after the W&B project. The resume cell then looks in `runs/detect/{WANDB_RUN_NAME}`, which is a different path.
- Log the split itself (the list of patient IDs per split) as a W&B artifact. That is what would have made §1.2 detectable.

### 3.4 Repo hygiene
- **~70+ MB of binaries are committed:** the base weights (`yolo11n.pt`, `yolo26m.pt` at 44 MB, `yolo26n.pt`, `yolov8n.pt`), every run's `best.pt` and `last.pt`, plus `.onnx` and `.torchscript`. They bloat the history permanently. Add `*.pt`, `*.onnx`, `*.torchscript` and `runs/` to `.gitignore` and keep models in W&B artifacts (you already upload them there). Consider `git rm --cached` for these files. Rewriting history is optional.
- `main.ipynb` is committed with outputs, which makes 3.5 MB diffs on every change. Use `nbstripout`, or clear outputs before committing.
- `yolo11n.pt` and `yolov8n.pt` appear to be leftovers. Delete them if unused.
- `requirements.txt` has no versions. Pin at least `ultralytics`, `albumentationsx` and `torch`. Ultralytics changes behaviour between releases.
- The README says YOLO26 is fine-tuned "with results", but it has no results, metrics table or example output. It also doesn't mention W&B setup.

---

## 4. What's genuinely good (keep doing this)

- **Thorough EDA:** class balance, the 3-class detail labels, age outliers (IQR), sex, view position, bbox size categories and a location heatmap.
- **Stratifying on `Target + view`** is a thoughtful choice. Most RSNA tutorials ignore view position.
- **Correct handling of multi-box images** and **empty label files for negatives**. Both are YOLO conventions people often get wrong.
- **Idempotent file generation** (skip files that already exist, keep image/label pairs in sync).
- **Augmenting train only.** The latest commit fixes a real val-contamination bug.
- **You already use experiment tracking**, write docstrings and explain your reasoning in comments.

---

## 5. Next steps (in order)

### Phase 0: Make evaluation trustworthy (do this first)
1. **Freeze a 3-way split**, stratified on `Target + view`: train / val / test, e.g. 70/15/15 by `patientId`. Save the ID lists to `splits/*.csv`, commit them, and log them to W&B. **Never look at test until the end.**
2. Regenerate `YOLO_format_data/` from those splits (as PNG). Delete the old images, labels and `*.cache`.
3. Build `model.train()` kwargs from a single config dict and log that dict.
4. Retrain a **quick baseline**: `yolo26n`, 640, built-in aug only (no offline aug), `hsv_s=0`, `hsv_h=0`, ~30 epochs, `patience=10`.
5. Evaluate: mAP50, mAP50-95, **image-level sensitivity/specificity/AUC**, and a sanity-checked confusion matrix from the same `val()` call.
6. Re-upload the W&B artifact with the correct metadata, and delete or relabel the mislabelled one.

### Phase 1: Controlled experiments (one variable at a time, on val)
| Experiment | Why |
|---|---|
| Offline aug vs. none | Check whether 3× positive oversampling actually helps (§2.1) |
| Negative ratio: all / 50% / 25% of negatives | Trade-off between recall and false positives |
| imgsz 512 / 768 / 1024 | Opacities are diffuse; find the sweet spot for your GPU |
| yolo26n → s → m | Capacity vs. overfitting (already overfits at n, so pair it with stronger aug) |
| CLAHE preprocessing | Cheap, common for CXR |
| `cos_lr=True`, `close_mosaic` tuning | Smoother convergence |
| TTA (`augment=True` at predict) | Usually a free gain for detection |

Record each one as its own W&B run, with the same split and seed.

### Phase 2: Match the real problem
1. **Implement the RSNA metric** (mean precision over IoU 0.4–0.75, per image) and **tune the confidence threshold** on val.
2. **Add an image-level classifier gate.** Many top RSNA solutions used a separate classifier, e.g. an EfficientNet/ConvNeXt on "pneumonia yes/no", to suppress boxes on healthy images. This often helps more than detector tweaks, because false positives on negatives are so costly.
3. **Ensemble** 2–3 detectors or folds with **Weighted Boxes Fusion** (`ensemble-boxes` package).
4. **Make a Kaggle late submission** on `stage_2_test_images` to get one externally comparable score.

### Phase 3: Analysis and presentation
1. Performance broken down by **view (AP/PA), sex and age group**.
2. A **predicted-box heatmap** vs. the ground-truth heatmap (the plan in cell 147).
3. **Error analysis:** look at the 20 worst false positives and false negatives. Are they label noise, small boxes, or AP views?
4. **Interpretability:** EigenCAM works with YOLO (Grad-CAM is awkward for detectors).
5. Results table and example predictions in the README. Optionally a small **Gradio demo** (upload an X-ray, get boxes) using the ONNX export.

### Phase 4: Clean-up (can run in parallel)
- Split `main.ipynb` into `src/` scripts plus the EDA and analysis notebooks (you've started with `eda.ipynb` and `model_training.ipynb`).
- Remove the weights and `runs/` from git, add `nbstripout`, and pin dependencies.
- Move the research notes (cells 145–157) out of the notebook into a `NOTES.md`.

### Later, optional
- Add external data: NIH ChestX-ray14 has some boxes; CheXpert and MIMIC-CXR have image-level labels and could pretrain the classifier gate.
- Feed the model view-position context (from your TODO list), e.g. separate models or thresholds per view, as a simpler first step than architecture changes.

---

## 6. Checklist

- [ ] 3-way split saved to `splits/` and logged to W&B
- [ ] Data regenerated as PNG, caches cleared
- [ ] Single config dict drives both training and logging
- [ ] Baseline retrained; mAP, image-level AUC/sens/spec and confusion matrix all agree
- [ ] Mislabelled W&B artifact fixed
- [ ] Offline-aug ablation done
- [ ] Image-size and model-size sweep done
- [ ] RSNA metric implemented and threshold tuned
- [ ] Classifier gate tried
- [ ] Test set evaluated **once**; results in README
- [ ] Weights and `runs/` out of git; deps pinned; notebook outputs stripped
