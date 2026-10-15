"""Shared helpers for the Simpsons classifier: data discovery, transforms, metrics, checkpoints."""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pretrained weights live inside the project: prefetch on the login node, use offline on GPU nodes.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_images(d: Path) -> bool:
    return any(p.suffix.lower() in IMG_EXT for p in d.iterdir() if p.is_file())


def descend(path: Path) -> Path:
    """Kaggle zips often nest folders (train/simpsons_dataset/..., testset/testset/...).
    Walk down while there is exactly one subfolder and no images."""
    path = Path(path)
    while True:
        subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(subdirs) == 1 and not has_images(path):
            path = subdirs[0]
        else:
            return path


def list_labeled(root: Path) -> tuple[list[str], list[tuple[Path, int]]]:
    root = descend(root)
    # A class folder holds images directly. This skips duplicate nested copies
    # like simpsons_dataset/simpsons_dataset that some Kaggle archives contain.
    classes = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and not d.name.startswith(".") and has_images(d))
    if not classes:
        raise SystemExit(f"No class folders in {root}. Expected <root>/<character>/*.jpg")
    samples = []
    for i, c in enumerate(classes):
        samples += [(p, i) for p in sorted((root / c).rglob("*")) if p.suffix.lower() in IMG_EXT]
    return classes, samples


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def list_unlabeled(root: Path) -> list[Path]:
    root = descend(root)
    return sorted((p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT), key=natural_key)


class ImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items  # [(path, label)] or [path]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        path, label = item if isinstance(item, tuple) else (item, -1)
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return x, label


def eval_transform(img: int, mean, std):
    return T.Compose([
        T.Resize(img, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def train_transform(img: int, mean, std):
    return T.Compose([
        T.RandomResizedCrop(img, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5)], p=0.3),
        # Small hue shift only: skin/hair color is a strong cue in this cartoon
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, value="random"),
    ])


def metrics(y_true: np.ndarray, y_pred: np.ndarray, nc: int) -> tuple[dict, np.ndarray, np.ndarray]:
    cm = np.bincount(nc * y_true + y_pred, minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    present = cm.sum(1) > 0
    acc = tp.sum() / max(1, cm.sum())
    # For single-label classification micro-F1 == accuracy
    return {"acc": acc, "f1_micro": acc, "f1_macro": float(f1[present].mean())}, cm, f1


@torch.no_grad()
def predict_probs(model, loader, device, tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        probs.append(p.cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def load_checkpoint(path: Path, device):
    import timm

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval().to(memory_format=torch.channels_last)
    return model, ckpt
