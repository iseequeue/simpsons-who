"""Fine-tune a pretrained timm backbone to classify Simpsons characters (Kaggle journey-springfield).

Recipe aimed at the leaderboard: ImageNet-22k ConvNeXt, AdamW + cosine with warmup,
Mixup/CutMix + label smoothing, mild class-balanced sampling, EMA of weights, fp16, flip TTA.

    # 1) train with a 10% stratified validation split, see the score and the worst classes
    python classification/train_cls.py --data data/journey/train --name convnext_s
    # 2) same recipe on 100% of the data for the final submission
    python classification/train_cls.py --data data/journey/train --name convnext_s_full --full

Output in runs_cls/<name>/: best.pt, classes.json, log.csv, report.txt, val_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (ROOT, ImageDataset, eval_transform, list_labeled, metrics, predict_probs,  # noqa: I001
                    train_transform)

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from timm.data import Mixup, resolve_data_config  # noqa: E402
from timm.loss import SoftTargetCrossEntropy  # noqa: E402
from timm.utils import ModelEmaV3  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with <character>/*.jpg (nesting is ok)")
    ap.add_argument("--name", default="convnext_s")
    ap.add_argument("--model", default="convnext_small.fb_in22k_ft_in1k_384")
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup epochs")
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.5, help="probability of mixup/cutmix per batch, 0 = off")
    ap.add_argument("--mixup-off", type=int, default=3, help="last N epochs without mixup")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay, 0 = off")
    ap.add_argument("--balance", type=float, default=0.5,
                    help="sampling weight = 1/count^balance (0 = natural, 1 = fully balanced)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--full", action="store_true", help="train on all data, no validation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="(tests only)")
    return ap.parse_args()


def stratified_split(samples, frac, seed):
    by_cls = defaultdict(list)
    for s in samples:
        by_cls[s[1]].append(s)
    rng = random.Random(seed)
    train, val = [], []
    for items in by_cls.values():
        rng.shuffle(items)
        n_val = max(1, round(len(items) * frac)) if len(items) >= 2 else 0
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def save_ckpt(path, model, args, classes, data_cfg, extra):
    torch.save({
        "model": args.model,
        "img": args.img,
        "mean": list(data_cfg["mean"]),
        "std": list(data_cfg["std"]),
        "classes": classes,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **extra,
    }, path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    out = ROOT / "runs_cls" / args.name
    out.mkdir(parents=True, exist_ok=True)

    classes, samples = list_labeled(args.data)
    nc = len(classes)
    train_items, val_items = (samples, []) if args.full else stratified_split(samples, args.val_frac, args.seed)
    print(f"{nc} classes, {len(samples)} images → train {len(train_items)}, val {len(val_items)}"
          f"{' (full mode)' if args.full else ''}")
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1))

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=nc,
                              drop_path_rate=args.drop_path)
    data_cfg = resolve_data_config({}, model=model)
    model.to(device).to(memory_format=torch.channels_last)
    print(f"model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, img {args.img}, "
          f"device {torch.cuda.get_device_name(0) if use_amp else 'cpu'}")

    # Mildly balanced sampling: rare characters (3–30 pictures) are seen more often
    counts = Counter(y for _, y in train_items)
    weights = [1.0 / counts[y] ** args.balance for _, y in train_items]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)

    loader_kw = dict(num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0)
    train_loader = DataLoader(ImageDataset(train_items, train_transform(args.img, data_cfg["mean"], data_cfg["std"])),
                              batch_size=args.bs, sampler=sampler, drop_last=True, **loader_kw)
    val_loader = None
    if val_items:
        val_loader = DataLoader(ImageDataset(val_items, eval_transform(args.img, data_cfg["mean"], data_cfg["std"])),
                                batch_size=args.bs * 2, shuffle=False, **loader_kw)

    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr)
    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    warm = max(1, int(args.warmup * steps_per_epoch))
    min_ratio = 0.01

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEmaV3(model, decay=args.ema, use_warmup=True) if args.ema > 0 else None

    mixup = None
    if args.mixup > 0:
        mixup = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=args.mixup, switch_prob=0.5,
                      label_smoothing=args.smoothing, num_classes=nc)
    soft_ce = SoftTargetCrossEntropy()
    hard_ce = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_acc", "val_f1_macro", "ema_acc", "ema_f1_macro", "sec"])

    best = {"score": -1.0}
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        use_mix = mixup is not None and epoch <= args.epochs - args.mixup_off
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                if use_mix:
                    x, y_soft = mixup(x, y)
                    loss = soft_ce(model(x), y_soft)
                else:
                    loss = hard_ce(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / max(1, n)}
        if val_loader is not None:
            for tag, m in (("val", model), ("ema", ema.module if ema else None)):
                if m is None:
                    continue
                probs, labels = predict_probs(m, val_loader, device, tta=False)
                res, _, _ = metrics(labels, probs.argmax(1), nc)
                row[f"{tag}_acc"], row[f"{tag}_f1_macro"] = res["acc"], res["f1_macro"]
                score = res["f1_macro"] + res["acc"]  # tie-break on accuracy
                if score > best["score"]:
                    best = {"score": score, "epoch": epoch, "which": tag, **res}
                    save_ckpt(out / "best.pt", m, args, classes, data_cfg,
                              {"epoch": epoch, "which": tag, "val": res})
        row["sec"] = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in
                                    ["epoch", "lr", "train_loss", "val_acc", "val_f1_macro", "ema_acc",
                                     "ema_f1_macro", "sec"]])
        msg = f"epoch {epoch:>3}/{args.epochs}  loss {row['train_loss']:.4f}  lr {row['lr']:.2e}  {row['sec']:.0f}s"
        if "val_acc" in row:
            msg += f"  | val acc {row['val_acc']:.4f} f1 {row['val_f1_macro']:.4f}"
        if "ema_acc" in row:
            msg += f"  | ema acc {row['ema_acc']:.4f} f1 {row['ema_f1_macro']:.4f}"
        print(msg, flush=True)

    # Full mode: no validation to pick from, EMA weights at the end are the usual choice
    final = ema.module if ema is not None else model
    save_ckpt(out / "last.pt", final, args, classes, data_cfg, {"epoch": args.epochs, "which": "ema" if ema else "raw"})
    if args.full:
        save_ckpt(out / "best.pt", final, args, classes, data_cfg, {"epoch": args.epochs, "which": "ema" if ema else "raw"})
        print(f"\nDone (full mode). Weights: {out / 'best.pt'}")
        return

    report(out, classes, val_items, val_loader, device, best)


def report(out, classes, val_items, val_loader, device, best):
    from common import load_checkpoint

    nc = len(classes)
    model, _ = load_checkpoint(out / "best.pt", device)
    probs, labels = predict_probs(model, val_loader, device, tta=True)
    pred = probs.argmax(1)
    res, cm, f1 = metrics(labels, pred, nc)
    counts = Counter(labels.tolist())

    lines = [f"Best: epoch {best['epoch']} ({best['which']} weights)",
             f"Validation with flip-TTA: accuracy = F1 micro = {res['acc']:.4f}, F1 macro = {res['f1_macro']:.4f}",
             f"Errors: {int((pred != labels).sum())} of {len(labels)}", "", "Worst classes (F1, val images):"]
    for i in np.argsort(f1)[:10]:
        if counts.get(i, 0):
            lines.append(f"  {classes[i]:<28} {f1[i]:.3f}  ({counts[i]})")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(off[i, j], i, j) for i in range(nc) for j in range(nc) if off[i, j]]
    if pairs:
        lines += ["", "Most common confusions (true → predicted):"]
        for c, i, j in sorted(pairs, reverse=True)[:10]:
            lines.append(f"  {classes[i]} → {classes[j]}: {c}")
    text = "\n".join(lines)
    print("\n" + text)
    (out / "report.txt").write_text(text + "\n")

    with open(out / "val_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence"])
        for (path, _), t, p, pr in zip(val_items, labels, pred, probs):
            if t != p:
                w.writerow([str(path), classes[t], classes[p], f"{pr[p]:.3f}"])
    print(f"\nWeights: {out / 'best.pt'}\nMistakes to look at: {out / 'val_errors.csv'}")


if __name__ == "__main__":
    main()
