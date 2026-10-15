"""Plots and metrics for a trained classifier (for the report / README).

    python classification/plots_cls.py --run runs_cls/convnext_s --data data/journey/train

Rebuilds the same validation split as train_cls.py (same seed and fraction), runs the
best checkpoint with flip-TTA and writes into runs_cls/<name>/:
    metrics.json, per_class.csv
    plots/curves.png              loss, accuracy, F1 macro, LR by epoch
    plots/class_distribution.png  images per character (imbalance)
    plots/per_class_f1.png        F1 per character with val support
    plots/confusion_matrix.png    full 42×42, row-normalized
    plots/confusion_errors.png    only the characters involved in mistakes, counts
    plots/confidence.png          confidence of right vs wrong answers + reliability diagram
    plots/errors_gallery.png      the misclassified pictures themselves
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from common import ImageDataset, eval_transform, list_labeled, load_checkpoint, predict_probs  # noqa: I001
from train_cls import stratified_split

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Palette: validated categorical slots 1–2, one-hue sequential blue, neutral chrome
SERIES_1 = "#2a78d6"   # blue: raw weights / correct / emphasis
SERIES_2 = "#eb6834"   # orange: EMA weights / wrong
MUTED_FILL = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "font.size": 10, "lines.linewidth": 2,
        "legend.frameon": False, "legend.labelcolor": INK_2, "savefig.dpi": 150,
    })


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ---------------------------------------------------------------- metrics

def compute_metrics(probs, labels, nc):
    pred = probs.argmax(1)
    cm = np.bincount(nc * labels + pred, minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = support > 0
    top3 = np.argsort(-probs, 1)[:, :3]
    conf = probs.max(1)
    correct = pred == labels
    # Expected calibration error, 15 equal-width bins
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 14)
    ece = sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
              for b in range(15) if (idx == b).any())
    summary = {
        "n_val": int(len(labels)),
        "errors": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "f1_micro": float(correct.mean()),
        "f1_macro": float(f1[present].mean()),
        "f1_weighted": float((f1 * support)[present].sum() / support.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "top3_accuracy": float((top3 == labels[:, None]).any(1).mean()),
        "log_loss": float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()),
        "ece": float(ece),
    }
    per_class = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return summary, per_class, cm


@torch.no_grad()
def speed(model, img, device):
    if device.type != "cuda":
        return {}
    out = {}
    for bs, iters in ((64, 20), (1, 50)):
        x = torch.randn(bs, 3, img, img, device=device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        if bs == 1:
            out["latency_ms_bs1"] = round(dt * 1000, 2)
        else:
            out["throughput_img_s_bs64"] = round(bs / dt, 1)
    out["gpu"] = torch.cuda.get_device_name(0)
    return out


# ---------------------------------------------------------------- plots

def plot_curves(log_csv: Path, out: Path):
    rows = list(csv.DictReader(open(log_csv)))
    ep = [int(r["epoch"]) for r in rows]
    get = lambda k: [float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    (a, b), (c, d) = axes

    a.plot(ep, get("train_loss"), color=SERIES_1)
    a.set_title("Train loss")

    for ax, key, title in ((b, "acc", "Validation accuracy"), (c, "f1_macro", "Validation F1 macro")):
        raw, ema = get(f"val_{key}"), get(f"ema_{key}")
        ax.plot(ep, raw, color=SERIES_1, label="model weights")
        ax.plot(ep, ema, color=SERIES_2, label="EMA weights")
        lo = np.nanmin(raw + ema)
        ax.set_ylim(max(lo - 0.005, 0), 1.0005)
        ax.set_title(title)
        best = int(np.nanargmax(ema)) if not np.all(np.isnan(ema)) else int(np.nanargmax(raw))
        val = max(ema[best], raw[best]) if not np.isnan(ema[best]) else raw[best]
        ax.annotate(f"{val:.4f}", (ep[best], val), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK_2, fontsize=9)
        ax.legend(loc="lower right")

    d.plot(ep, get("lr"), color=SERIES_1)
    d.set_title("Learning rate (end of epoch)")
    d.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in (c, d):
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_distribution(counts: Counter, classes, out: Path):
    order = sorted(range(len(classes)), key=lambda i: counts[i])
    fig, ax = plt.subplots(figsize=(8, 0.24 * len(classes) + 1.2))
    vals = [counts[i] for i in order]
    ax.barh(range(len(order)), vals, height=0.75, color=SERIES_1)
    ax.set_yticks(range(len(order)), [pretty(classes[i]) for i in order], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(left=0.7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("images (log scale)")
    ax.set_title(f"Images per character: {min(vals)} to {max(vals)}")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(vals):
        ax.text(v * 1.05, y, str(v), va="center", fontsize=7, color=INK_2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_per_class(per_class, classes, out: Path):
    f1, sup = per_class["f1"], per_class["support"]
    order = [i for i in sorted(range(len(classes)), key=lambda i: (f1[i], sup[i]), reverse=True) if sup[i] > 0]
    fig, ax = plt.subplots(figsize=(8, 0.24 * len(order) + 1.2))
    colors = [SERIES_1 if f1[i] < 1 else MUTED_FILL for i in order]
    ax.barh(range(len(order)), [f1[i] for i in order], height=0.75, color=colors)
    ax.set_yticks(range(len(order)), [f"{pretty(classes[i])}  (n={sup[i]})" for i in order], fontsize=8)
    lo = min(f1[i] for i in order)
    ax.set_xlim(max(0, lo - 0.05), 1.0)
    ax.set_xlabel("F1 on validation")
    n_imperfect = sum(f1[i] < 1 for i in order)
    ax.set_title(f"F1 per character: {n_imperfect} of {len(order)} below 1.0 (in blue)")
    ax.grid(axis="y", visible=False)
    for y, i in enumerate(order):
        if f1[i] < 1:
            ax.text(f1[i] - 0.002, y, f"{f1[i]:.3f}", va="center", ha="right", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(cm, classes, out_full: Path, out_err: Path):
    nc = len(classes)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(nc), [pretty(c) for c in classes], rotation=90, fontsize=7)
    ax.set_yticks(range(nc), [pretty(c) for c in classes], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title("Confusion matrix (share of each true character)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_full)
    plt.close(fig)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    involved = sorted(set(np.nonzero(off.sum(1))[0]) | set(np.nonzero(off.sum(0))[0]))
    if not involved:
        return
    sub = cm[np.ix_(involved, involved)]
    k = len(involved)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * k + 2.5), max(4, 0.5 * k + 2)))
    show = np.where(np.eye(k, dtype=bool), 0, sub)  # color only the mistakes
    ax.imshow(show, cmap=BLUES, vmin=0, vmax=max(1, show.max()))
    for r in range(k):
        for c_ in range(k):
            v = sub[r, c_]
            if v:
                diag = r == c_
                ax.text(c_, r, str(v), ha="center", va="center", fontsize=8,
                        color=INK_MUTED if diag else ("white" if v > show.max() / 2 else INK))
    names = [pretty(classes[i]) for i in involved]
    ax.set_xticks(range(k), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    n_err = int(off.sum())
    ax.set_title(f"Where the {n_err} mistake{'s' if n_err != 1 else ''} are (gray numbers = correct)"
                 .replace("mistake are", "mistake is"))
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_err)
    plt.close(fig)


def plot_confidence(probs, labels, out: Path, ece: float):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 1, 21)
    a.hist(conf[correct], bins=bins, histtype="step", color=SERIES_1, linewidth=2,
           label=f"correct ({correct.sum()})")
    if (~correct).any():
        a.hist(conf[~correct], bins=bins, histtype="step", color=SERIES_2, linewidth=2,
               label=f"wrong ({(~correct).sum()})")
    a.set_yscale("log")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("model confidence (top probability)")
    a.set_ylabel("images (log scale)")
    a.set_title("Confidence of right and wrong answers")
    a.legend(loc="upper left")

    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for i in range(10):
        m = idx == i
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    b.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    b.plot(xs, ys, color=SERIES_1, marker="o", markersize=8)
    for x, y, n in zip(xs, ys, ns):
        b.annotate(f"n={n}", (x, y), xytext=(0, -16), textcoords="offset points", ha="center",
                   fontsize=8, color=INK_2)
    b.set_xlim(0, 1.02)
    b.set_ylim(0, 1.05)
    b.set_xlabel("mean confidence in bin")
    b.set_ylabel("accuracy in bin")
    b.set_title(f"Reliability diagram (ECE {ece:.4f})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_errors(val_items, probs, labels, classes, out: Path, max_n=24):
    pred = probs.argmax(1)
    wrong = np.nonzero(pred != labels)[0]
    if not len(wrong):
        return
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:max_n]  # most confident mistakes first
    cols = min(6, len(wrong))
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(max(8.0, 2.6 * cols), 3.1 * rows + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        with Image.open(val_items[i][0]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{pretty(classes[labels[i]])}\n→ {pretty(classes[pred[i]])} ({probs[i, pred[i]]:.2f})",
                     fontsize=8, color=INK, loc="center", fontweight="normal")
    fig.suptitle(f"Validation mistakes: {len(np.nonzero(pred != labels)[0])} (true → predicted, confidence)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="runs_cls/<name>")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.run / "best.pt", device)
    classes = meta["classes"]
    data_classes, samples = list_labeled(args.data)
    if data_classes != classes:
        raise SystemExit("Class list in the data differs from the checkpoint")
    if meta.get("val") is None:
        print("⚠ This checkpoint was trained with --full (no hold-out): validation numbers will be optimistic")
    _, val_items = stratified_split(samples, args.val_frac, args.seed)

    loader = DataLoader(ImageDataset(val_items, eval_transform(meta["img"], meta["mean"], meta["std"])),
                        batch_size=args.bs, num_workers=args.workers, pin_memory=device.type == "cuda")
    probs, labels = predict_probs(model, loader, device, tta=not args.no_tta)
    nc = len(classes)
    summary, per_class, cm = compute_metrics(probs, labels, nc)

    ckpt_mb = (args.run / "best.pt").stat().st_size / 1e6
    summary.update({
        "model": meta["model"], "img": meta["img"], "tta": not args.no_tta,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "checkpoint_mb": round(ckpt_mb, 1), **speed(model, meta["img"], device),
    })
    (args.run / "metrics.json").write_text(json.dumps(summary, indent=2))
    with open(args.run / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "val_support", "total_images"])
        totals = Counter(y for _, y in samples)
        for i, c in enumerate(classes):
            w.writerow([c, f"{per_class['precision'][i]:.4f}", f"{per_class['recall'][i]:.4f}",
                        f"{per_class['f1'][i]:.4f}", int(per_class["support"][i]), totals[i]])

    plots = args.run / "plots"
    plots.mkdir(exist_ok=True)
    if (args.run / "log.csv").exists():
        plot_curves(args.run / "log.csv", plots / "curves.png")
    plot_distribution(Counter(y for _, y in samples), classes, plots / "class_distribution.png")
    plot_per_class(per_class, classes, plots / "per_class_f1.png")
    plot_confusion(cm, classes, plots / "confusion_matrix.png", plots / "confusion_errors.png")
    plot_confidence(probs, labels, plots / "confidence.png", summary["ece"])
    plot_errors(val_items, probs, labels, classes, plots / "errors_gallery.png")

    print(json.dumps(summary, indent=2))
    print(f"\nPlots: {plots}\nPer-class table: {args.run / 'per_class.csv'}")


if __name__ == "__main__":
    main()
