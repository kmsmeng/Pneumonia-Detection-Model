"""Build the YOLO dataset in data/yolo/ from the frozen splits.

- Images: DICOM pixel data saved as lossless 8-bit grayscale PNG at full resolution (1024x1024).
- Labels: one .txt per image, `0 x_center y_center width height` (normalized). Negatives get an EMPTY file.
- data.yaml with train/val/test.

No offline augmentation: Ultralytics' built-in augmentation is used at train time (see configs/).

    python src/prepare_data.py            # create (skips files that already exist)
    python src/prepare_data.py --clean    # delete data/yolo first and rebuild from scratch
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml
from pydicom import dcmread
from tqdm import tqdm

from common import SPLITS, YOLO_DIR, dcm_path, load_boxes, load_split


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    if pixels.dtype == np.uint8:
        return pixels
    p = pixels.astype(np.float32)
    p = (p - p.min()) / max(float(p.max() - p.min()), 1e-6) * 255.0
    return p.astype(np.uint8)


def yolo_lines(boxes, rows: int, cols: int) -> list[str]:
    lines = []
    for b in boxes.itertuples(index=False):
        x0, y0 = max(b.x, 0.0), max(b.y, 0.0)
        x1, y1 = min(b.x + b.width, cols), min(b.y + b.height, rows)
        xc, yc = (x0 + x1) / 2 / cols, (y0 + y1) / 2 / rows
        w, h = (x1 - x0) / cols, (y1 - y0) / rows
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help=f"delete {YOLO_DIR} before building")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.clean and YOLO_DIR.exists():
        print(f"Deleting {YOLO_DIR} ...")
        shutil.rmtree(YOLO_DIR)

    boxes = load_boxes()
    boxes_by_id = dict(tuple(boxes.groupby("patientId")))

    for split in SPLITS:
        df = load_split(split)
        img_dir, lbl_dir = YOLO_DIR / "images" / split, YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        def build(row) -> None:
            pid = row.patientId
            img_path, lbl_path = img_dir / f"{pid}.png", lbl_dir / f"{pid}.txt"
            if not img_path.exists():
                pixels = to_uint8(dcmread(dcm_path(pid)).pixel_array)
                tmp = img_path.with_suffix(".tmp.png")
                if not cv2.imwrite(str(tmp), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f"failed to write {tmp}")
                tmp.replace(img_path)  # atomic: an interrupted run never leaves a half-written PNG
            if not lbl_path.exists():
                b = boxes_by_id.get(pid)
                lines = [] if b is None else yolo_lines(b, row.rows, row.cols)
                lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(build, df.itertuples(index=False)), total=len(df), desc=split))

        # sanity check: label files agree with the split CSV
        n_pos = sum(1 for p in lbl_dir.glob("*.txt") if p.stat().st_size > 0)
        n_img = sum(1 for _ in img_dir.glob("*.png"))
        assert n_img == len(df), f"{split}: {n_img} images on disk but {len(df)} in split (stale files?)"
        assert n_pos == int(df.Target.sum()), f"{split}: {n_pos} positive label files, expected {df.Target.sum()}"
        print(f"  {split}: {n_img} images, {n_pos} positive")

    for cache in (YOLO_DIR / "labels").glob("*.cache"):
        cache.unlink()  # Ultralytics label caches go stale when labels change

    data_yaml = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": 1, "names": ["pneumonia"]}
    (YOLO_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"Wrote {YOLO_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
