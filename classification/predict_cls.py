"""Predict the Kaggle test set and write a submission. Several checkpoints = ensemble (averaged probabilities).

    python classification/predict_cls.py \
        --ckpt runs_cls/convnext_s_full/best.pt \
        --test-dir data/journey/testset \
        --sample-submission data/journey/sample_submission.csv \
        --out submission.csv

    # ensemble of several models (each with its own image size)
    python classification/predict_cls.py --ckpt runs_cls/a/best.pt runs_cls/b/best.pt --test-dir ... --out sub.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ImageDataset, eval_transform, list_unlabeled, load_checkpoint, predict_probs  # noqa: I001

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--sample-submission", type=Path, default=None,
                    help="Kaggle sample_submission.csv: column names and Id order are taken from it")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = list_unlabeled(args.test_dir)
    if not files:
        raise SystemExit(f"No images in {args.test_dir}")
    by_name = {p.name: p for p in files}

    id_col, label_col, ids = "Id", "Expected", [p.name for p in files]
    if args.sample_submission and args.sample_submission.exists():
        with open(args.sample_submission, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            id_col, label_col = header[0], header[1]
            ids = [row[0] for row in reader if row]
        missing = [i for i in ids if i not in by_name]
        if missing:
            raise SystemExit(f"{len(missing)} ids from the sample submission have no image, e.g. {missing[:3]}")
        print(f"Order and columns from {args.sample_submission}: {id_col},{label_col} ({len(ids)} rows)")
    elif args.sample_submission:
        print(f"⚠ {args.sample_submission} not found: writing Id,Expected for {len(ids)} files in name order")
    paths = [by_name[i] for i in ids]

    total, classes = None, None
    for ck in args.ckpt:
        model, meta = load_checkpoint(ck, device)
        if classes is None:
            classes = meta["classes"]
        elif meta["classes"] != classes:
            raise SystemExit(f"{ck} was trained on a different class list, can't ensemble")
        loader = DataLoader(ImageDataset(paths, eval_transform(meta["img"], meta["mean"], meta["std"])),
                            batch_size=args.bs, num_workers=args.workers, pin_memory=device.type == "cuda")
        probs, _ = predict_probs(model, loader, device, tta=not args.no_tta)
        total = probs if total is None else total + probs
        val = meta.get("val")
        note = f", val acc {val['acc']:.4f} / F1 macro {val['f1_macro']:.4f}" if val else ""
        print(f"  {ck}: {meta['model']} @ {meta['img']}px{note}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = total / len(args.ckpt)
    pred = probs.argmax(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([id_col, label_col])
        for i, p in zip(ids, pred):
            w.writerow([i, classes[p]])
    np.save(args.out.with_suffix(".probs.npy"), probs)

    conf = probs.max(1)
    print(f"\n{args.out}: {len(ids)} predictions")
    print(f"low-confidence (<0.5): {(conf < 0.5).sum()}, mean confidence {conf.mean():.3f}")


if __name__ == "__main__":
    main()
