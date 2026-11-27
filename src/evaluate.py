"""Evaluate a trained run on a split. Reports box-level metrics (Ultralytics) AND image-level metrics
("does this X-ray have pneumonia?"), then cross-checks that the numbers are mutually consistent.

    python src/evaluate.py runs/detect/baseline_n640                 # val split
    python src/evaluate.py runs/detect/baseline_n640 --split test --confirm-test

Image-level score = highest box confidence in the image (0 if no box). Results are written to
<run>/eval_<split>.json (per-image scores to <run>/eval_<split>_scores.csv) and, if the run has a
wandb_id.txt, added to that W&B run's summary.

Threshold analysis: on val, it picks operating points (thresholds for 90% / 80% sensitivity and the
best-F1 threshold), overall and per view. On test, it applies those val-chosen thresholds unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

from common import YOLO_DIR, load_config, load_split, resolve, splits_fingerprint


def image_scores(model, image_paths: list[Path], imgsz: int, batch: int) -> np.ndarray:
    """Highest box confidence per image (0 if no box). A list of paths is inferred as one batch."""
    scores = np.zeros(len(image_paths), dtype=np.float32)
    for start in tqdm(range(0, len(image_paths), batch), desc="image-level predictions", unit="batch"):
        chunk = [str(p) for p in image_paths[start:start + batch]]
        results = model.predict(chunk, imgsz=imgsz, conf=0.001, verbose=False)
        for i, r in enumerate(results):
            if len(r.boxes):
                scores[start + i] = float(r.boxes.conf.max())
    return scores


def threshold_metrics(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = s >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sens, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return {
        "threshold": float(t),
        "sensitivity": sens,
        "specificity": tn / max(tn + fp, 1),
        "precision": prec,
        "f1": 2 * prec * sens / max(prec + sens, 1e-12),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


SENS_TARGETS = (0.90, 0.80)


def operating_points(y: np.ndarray, s: np.ndarray) -> dict:
    """Pick thresholds on THIS data: the highest threshold reaching each target sensitivity
    (= best specificity for that sensitivity), and the threshold with the best F1."""
    _, tpr, thr = roc_curve(y, s)  # thresholds descending, tpr ascending
    points = {}
    for target in SENS_TARGETS:
        i = int(np.argmax(tpr >= target))
        points[f"sens{round(target * 100)}"] = threshold_metrics(y, s, thr[i])
    prec, rec, thr_pr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    points["best_f1"] = threshold_metrics(y, s, thr_pr[int(np.argmax(f1))])
    return points


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

    # Thresholds are chosen on val only. On test, the val-chosen thresholds are applied unchanged
    # (choosing them on test would inflate the final result).
    val_points = None
    if args.split == "test":
        val_file = run_dir / "eval_val.json"
        val_res = json.loads(val_file.read_text()) if val_file.exists() else {}
        if val_res.get("weights") != args.weights or "operating_points" not in val_res.get("image", {}):
            raise SystemExit(f"Run the val evaluation first (with --weights {args.weights}): "
                             "test uses the thresholds chosen on val.")
        val_points = {"all": val_res["image"]["operating_points"],
                      **{v: r["operating_points"] for v, r in val_res["image_by_view"].items()}}

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
    scores_file = run_dir / f"eval_{args.split}_scores.csv"
    df[["patientId", "view", "Target"]].assign(score=s).to_csv(scores_file, index=False)

    def summarize(mask, group: str) -> dict:
        yy, ss = y[mask], s[mask]
        if val_points is None:
            points = operating_points(yy, ss)
        else:
            points = {k: threshold_metrics(yy, ss, p["threshold"]) for k, p in val_points[group].items()}
        return {"n": int(mask.sum()), "positives": int(yy.sum()),
                "roc_auc": float(roc_auc_score(yy, ss)), "pr_auc": float(average_precision_score(yy, ss)),
                **threshold_metrics(yy, ss, args.conf), "operating_points": points}

    image = summarize(np.ones_like(y, dtype=bool), "all")
    by_view = {v: summarize((df.view == v).to_numpy(), v) for v in sorted(df.view.unique())}

    # Consistency checks: catch evaluation bugs like the old "TP = 0 but mAP50 = 0.25" matrix.
    warnings = []
    n_gt = int(df.n_boxes.sum())
    if box["cm_tp"] + box["cm_fn"] != n_gt:
        warnings.append(f"confusion matrix has {box['cm_tp'] + box['cm_fn']} GT boxes, split CSV has {n_gt}")
    if box["mAP50"] > 0.05 and box["cm_tp"] == 0:
        warnings.append("mAP50 > 0 but confusion-matrix TP == 0: matrix and metrics disagree")

    result = {"run": run_dir.name, "split": args.split, "weights": args.weights, "imgsz": imgsz,
              "split_fingerprint": splits_fingerprint(),
              "thresholds_chosen_on": "val",
              "box": box, "image": image, "image_by_view": by_view, "warnings": warnings}
    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== {run_dir.name} on {args.split} ===")
    print(f"box   mAP50 {box['mAP50']:.4f}  mAP50-95 {box['mAP50-95']:.4f}  P {box['precision']:.3f}  R {box['recall']:.3f}")
    print(f"      confusion (conf 0.25, IoU 0.45): TP {box['cm_tp']}  FP {box['cm_fp']}  FN {box['cm_fn']}  (GT boxes {n_gt})")
    print(f"image ROC-AUC {image['roc_auc']:.4f}  PR-AUC {image['pr_auc']:.4f}  @conf {args.conf}: "
          f"sens {image['sensitivity']:.3f}  spec {image['specificity']:.3f}")
    for v, r in by_view.items():
        print(f"      {v}: n={r['n']}  ROC-AUC {r['roc_auc']:.4f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")

    print(f"\nOperating points (thresholds chosen on val{', applied unchanged to test' if val_points else ''}):")
    print(f"  {'group':<5} {'point':<8} {'thresh':>7} {'sens':>6} {'spec':>6} {'prec':>6} {'F1':>6} {'FN':>5} {'FP':>5}")
    for group, r in {"all": image, **by_view}.items():
        for name, p in r["operating_points"].items():
            print(f"  {group:<5} {name:<8} {p['threshold']:>7.4f} {p['sensitivity']:>6.3f} {p['specificity']:>6.3f} "
                  f"{p['precision']:>6.3f} {p['f1']:>6.3f} {p['fn']:>5} {p['fp']:>5}")
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"Saved {out} and {scores_file.name}")

    id_file = run_dir / "wandb_id.txt"
    if not args.no_wandb and id_file.exists():
        import wandb
        run = wandb.init(project=cfg.get("wandb_project", "rsna-pneumonia-detection"), id=id_file.read_text().strip(),
                         resume="must")
        flat = {f"eval/{args.split}/box/{k}": v for k, v in box.items()}
        flat |= {f"eval/{args.split}/image/{k}": v for k, v in image.items() if k != "operating_points"}
        for group, r in {"all": image, **by_view}.items():
            for name, p in r["operating_points"].items():
                flat |= {f"eval/{args.split}/image/{group}/{name}/{k}": v for k, v in p.items()}
        run.summary.update(flat)
        run.finish()


if __name__ == "__main__":
    main()
