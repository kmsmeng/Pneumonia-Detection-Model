"""Create the frozen train/val/test split (by patientId) and write it to splits/*.csv.

Stratified on Target + ViewPosition. Each CSV row is one patient/image, with the DICOM metadata
needed for later breakdowns (view, age, sex).

The split is meant to be created ONCE and committed. Re-running refuses to overwrite unless --force.

    python src/make_splits.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pydicom import dcmread
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import RAW_CLASS_INFO_CSV, RAW_LABELS_CSV, SPLITS, SPLITS_DIR, dcm_path


def read_meta(patient_id: str) -> dict:
    ds = dcmread(dcm_path(patient_id), stop_before_pixels=True)
    return {
        "patientId": patient_id,
        "view": ds.ViewPosition,
        "age": int(ds.PatientAge),
        "sex": ds.PatientSex,
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    existing = [SPLITS_DIR / f"{s}.csv" for s in SPLITS if (SPLITS_DIR / f"{s}.csv").exists()]
    if existing and not args.force:
        raise SystemExit(
            f"Split files already exist ({', '.join(p.name for p in existing)}). The split is frozen; "
            "pass --force only if you really mean to invalidate every model trained on it."
        )

    labels = pd.read_csv(RAW_LABELS_CSV)
    patients = (
        labels.groupby("patientId", sort=False)
        .agg(Target=("Target", "first"), n_boxes=("x", "count"))
        .reset_index()
    )
    classes = pd.read_csv(RAW_CLASS_INFO_CSV).drop_duplicates("patientId")
    patients = patients.merge(classes, on="patientId", how="left")

    with ThreadPoolExecutor(max_workers=16) as ex:
        meta = list(tqdm(ex.map(read_meta, patients["patientId"]), total=len(patients), desc="DICOM headers"))
    patients = patients.merge(pd.DataFrame(meta), on="patientId", how="left")

    strat = patients["Target"].astype(str) + "_" + patients["view"]
    trainval, test = train_test_split(patients, test_size=args.test_size, stratify=strat, random_state=args.seed)
    strat_tv = trainval["Target"].astype(str) + "_" + trainval["view"]
    val_frac = args.val_size / (1 - args.test_size)
    train, val = train_test_split(trainval, test_size=val_frac, stratify=strat_tv, random_state=args.seed)

    assert not (set(train.patientId) & set(val.patientId) | set(train.patientId) & set(test.patientId)
                | set(val.patientId) & set(test.patientId)), "patient overlap between splits"

    SPLITS_DIR.mkdir(exist_ok=True)
    cols = ["patientId", "Target", "n_boxes", "class", "view", "age", "sex", "rows", "cols"]
    print(f"\n{'split':<6} {'images':>7} {'pos':>6} {'pos%':>6} {'boxes':>6} {'AP%':>5}")
    for name, df in zip(SPLITS, (train, val, test)):
        df = df.sort_values("patientId")[cols]
        df.to_csv(SPLITS_DIR / f"{name}.csv", index=False, lineterminator="\n")
        print(f"{name:<6} {len(df):>7} {df.Target.sum():>6} {df.Target.mean() * 100:>5.1f}% "
              f"{df.n_boxes.sum():>6} {(df.view == 'AP').mean() * 100:>4.1f}%")
    print(f"\nWrote {SPLITS_DIR}. Commit these files.")


if __name__ == "__main__":
    main()
