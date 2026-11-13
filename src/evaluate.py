"""Evaluate a trained run on a split. Reports box-level metrics (Ultralytics) AND image-level metrics
("does this X-ray have pneumonia?"), then cross-checks that the numbers are mutually consistent.

    python src/evaluate.py runs/detect/baseline_n640                 # val split
    python src/evaluate.py runs/detect/baseline_n640 --split test --confirm-test

Image-level score = highest box confidence in the image (0 if no box). Results are written to
<run>/eval_<split>.json and, if the run has a wandb_id.txt, added to that W&B run's summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import YOLO_DIR, load_config, load_split, resolve, splits_fingerprint


def image_scores(model, image_paths: list[Path], imgsz: int, batch: int) -> np.ndarray:
    scores = np.zeros(len(image_paths), dtype=np.float32)
    for start in range(0, len(image_paths), batch):
        chunk = [str(p) for p in image_paths[start:start + batch]]
        for i, r in enumerate(model.predict(chunk, imgsz=imgsz, conf=0.001, verbose=False)):
            if len(r.boxes):
                scores[start + i] = float(r.boxes.conf.max())
    return scores


def threshold_metrics(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = s >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {
        "threshold": round(float(t), 4),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory, e.g. runs/detect/baseline_n640")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-test", action="store_true", help="required for --split test (use it ONCE, at the end)")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold for image-level sens/spec")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test split is for the final, one-time evaluation. Re-run with --confirm-test if that's now.")

    from ultralytics import YOLO

    run_dir = resolve(args.run)
    cfg = load_config(run_dir / "config.yaml")
    imgsz = cfg.get("train", {}).get("imgsz", 640)

    prov_file = run_dir / "provenance.json"
    if prov_file.exists():
        trained_on = json.loads(prov_file.read_text())["split_fingerprint"]
        if trained_on != splits_fingerprint():
            raise SystemExit(f"Run was trained on split {trained_on}, but splits/ is now {splits_fingerprint()}. "
                             "Evaluating it on the current split would leak training data.")

    model = YOLO(run_dir / "weights" / args.weights)

    # Box-level metrics (Ultralytics). plots=True is required for the confusion matrix to be filled.
    m = model.val(data=str(resolve(cfg["data"])), split=args.split, imgsz=imgsz, batch=args.batch,
                  plots=True, project=str(run_dir), name=f"eval_{args.split}", exist_ok=True, verbose=False)
    cm = m.confusion_matrix.matrix  # indexed [predicted, true]; last index = background
    box = {
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "cm_tp": int(cm[0, 0]), "cm_fp": int(cm[0, 1]), "cm_fn": int(cm[1, 0]),
    }

    # Image-level metrics
    df = load_split(args.split)
    paths = [YOLO_DIR / "images" / args.split / f"{pid}.png" for pid in df.patientId]
    y = df.Target.to_numpy()
    s = image_scores(model, paths, imgsz, args.batch)

    def summarize(mask) -> dict:
        yy, ss = y[mask], s[mask]
        return {"n": int(mask.sum()), "positives": int(yy.sum()),
                "roc_auc": float(roc_auc_score(yy, ss)), "pr_auc": float(average_precision_score(yy, ss)),
                **threshold_metrics(yy, ss, args.conf)}

    image = summarize(np.ones_like(y, dtype=bool))
    by_view = {v: summarize((df.view == v).to_numpy()) for v in sorted(df.view.unique())}

    # Consistency checks: catch evaluation bugs like the old "TP = 0 but mAP50 = 0.25" matrix.
    warnings = []
    n_gt = int(df.n_boxes.sum())
    if box["cm_tp"] + box["cm_fn"] != n_gt:
        warnings.append(f"confusion matrix has {box['cm_tp'] + box['cm_fn']} GT boxes, split CSV has {n_gt}")
    if box["mAP50"] > 0.05 and box["cm_tp"] == 0:
        warnings.append("mAP50 > 0 but confusion-matrix TP == 0: matrix and metrics disagree")

    result = {"run": run_dir.name, "split": args.split, "weights": args.weights, "imgsz": imgsz,
              "split_fingerprint": splits_fingerprint(), "box": box, "image": image, "image_by_view": by_view,
              "warnings": warnings}
    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== {run_dir.name} on {args.split} ===")
    print(f"box   mAP50 {box['mAP50']:.4f}  mAP50-95 {box['mAP50-95']:.4f}  P {box['precision']:.3f}  R {box['recall']:.3f}")
    print(f"      confusion (conf 0.25, IoU 0.45): TP {box['cm_tp']}  FP {box['cm_fp']}  FN {box['cm_fn']}  (GT boxes {n_gt})")
    print(f"image ROC-AUC {image['roc_auc']:.4f}  PR-AUC {image['pr_auc']:.4f}  @conf {args.conf}: "
          f"sens {image['sensitivity']:.3f}  spec {image['specificity']:.3f}")
    for v, r in by_view.items():
        print(f"      {v}: n={r['n']}  ROC-AUC {r['roc_auc']:.4f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"Saved {out}")

    id_file = run_dir / "wandb_id.txt"
    if not args.no_wandb and id_file.exists():
        import wandb
        run = wandb.init(project=cfg.get("wandb_project", "rsna-pneumonia-detection"), id=id_file.read_text().strip(),
                         resume="must")
        flat = {f"eval/{args.split}/box/{k}": v for k, v in box.items()}
        flat |= {f"eval/{args.split}/image/{k}": v for k, v in image.items()}
        run.summary.update(flat)
        run.finish()


if __name__ == "__main__":
    main()
