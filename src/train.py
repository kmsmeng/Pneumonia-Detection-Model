"""Train a YOLO model from a config file. The config is the single source of truth: it is passed
straight to `model.train()`, logged to W&B as-is, and copied into the run directory.

    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/baseline.yaml --no-wandb
    python src/train.py --config configs/baseline.yaml --set workers=8 batch=32   # per-machine overrides
    python src/train.py --resume runs/detect/baseline_n640        # continue an interrupted run

`--set key=value` overrides a key under `train:` (values parsed as YAML, so 8 -> int, true -> bool).
The merged result is what gets saved to the run's config.yaml and logged, so the record always matches the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common import RUNS_DIR, git_commit, load_config, resolve, splits_fingerprint


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)  # PyYAML (YAML 1.1) reads "5e-4" as a string
            except ValueError:
                pass
        overrides[key.strip()] = parsed
    return overrides


def init_wandb(cfg: dict, run_dir: Path, resume_id: str | None = None):
    import wandb

    run = wandb.init(
        project=cfg.get("wandb_project", "rsna-pneumonia-detection"),
        name=cfg["name"],
        id=resume_id,
        resume="must" if resume_id else None,
        job_type="train",
        config=None if resume_id else {
            "config": cfg,
            "split_fingerprint": splits_fingerprint(),
            "git_commit": git_commit(),
        },
    )
    (run_dir / "wandb_id.txt").write_text(run.id, encoding="utf-8")
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", type=Path)
    g.add_argument("--resume", type=Path, help="run directory to resume")
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="override keys under `train:` in the config, e.g. --set workers=8 batch=32")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    if args.resume and args.set:
        raise SystemExit("--set can't be combined with --resume (a resumed run keeps its original settings).")

    from ultralytics import YOLO, settings

    settings.update({"wandb": not args.no_wandb})  # Ultralytics' W&B callback reuses the run we start below

    if args.resume:
        run_dir = resolve(args.resume)
        cfg = load_config(run_dir / "config.yaml")
        if not args.no_wandb:
            id_file = run_dir / "wandb_id.txt"
            init_wandb(cfg, run_dir, id_file.read_text().strip() if id_file.exists() else None)
        YOLO(run_dir / "weights" / "last.pt").train(resume=True)
        return

    cfg = load_config(args.config)
    overrides = parse_overrides(args.set)
    cfg.setdefault("train", {}).update(overrides)

    run_dir = RUNS_DIR / cfg["name"]
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists. Change `name` in the config, or use --resume {run_dir}.")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "provenance.json").write_text(
        json.dumps({"split_fingerprint": splits_fingerprint(), "git_commit": git_commit(),
                    "config_file": str(args.config), "overrides": overrides}, indent=2),
        encoding="utf-8",
    )

    if not args.no_wandb:
        init_wandb(cfg, run_dir)

    model = YOLO(resolve(cfg["model"]))
    model.train(
        data=str(resolve(cfg["data"])),
        project=str(RUNS_DIR),  # absolute, so Ultralytics doesn't nest it under its global runs_dir
        name=cfg["name"],
        exist_ok=True,          # we created run_dir ourselves above
        **cfg.get("train", {}),
    )


if __name__ == "__main__":
    main()
