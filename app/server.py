"""Simpsons character classifier: web service (FastAPI + ONNX Runtime, no PyTorch needed).

    cd app && uvicorn server:app --host 0.0.0.0 --port 8000      # then open http://localhost:8000

API:
    POST /api/predict   multipart: image=<file>, box="x1,y1,x2,y2" (optional, pixels of the original)
    GET  /api/info      model name, classes, validation metrics
    GET  /health
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "model"))
MAX_BYTES = 15 * 2**20
CONFIDENT = float(os.environ.get("CONFIDENT", 0.5))  # below this the page says "not sure"
ALSO_MIN = float(os.environ.get("ALSO_MIN", 0.6))    # another character in a different part of the frame
MAX_SIDE = 4096    # screenshots bigger than this are downscaled first

META = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
IMG = int(META["img"])
MEAN = np.array(META["mean"], np.float32).reshape(3, 1, 1)
STD = np.array(META["std"], np.float32).reshape(3, 1, 1)
CLASSES: list[str] = META["classes"]

_opts = ort.SessionOptions()
_opts.intra_op_num_threads = int(os.environ.get("THREADS", 0))  # 0 = all cores
SESSION = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), _opts, providers=["CPUExecutionProvider"])
INPUT = SESSION.get_inputs()[0].name

app = FastAPI(title="Simpsons character classifier")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def preprocess(im: Image.Image) -> np.ndarray:
    """Exactly torchvision Resize(IMG, bicubic) → CenterCrop(IMG) → ToTensor → Normalize (as in training)."""
    w, h = im.size
    if w <= h:
        nw, nh = IMG, int(IMG * h / w)
    else:
        nw, nh = int(IMG * w / h), IMG
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = int(round((nw - IMG) / 2.0)), int(round((nh - IMG) / 2.0))
    im = im.crop((left, top, left + IMG, top + IMG))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - MEAN) / STD


def windows(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Square views that together cover a wide (or tall) screenshot: left / center / right.
    The classifier was trained on fairly tight shots, so looking at parts finds side characters
    and tells where the answer came from."""
    short, long = min(w, h), max(w, h)
    if long / short <= 1.2:
        return [(0, 0, w, h)]
    n = math.ceil(long / short) + 1
    offs = [round(i * (long - short) / (n - 1)) for i in range(n)]
    return [(o, 0, o + short, h) if w > h else (0, o, w, o + short) for o in offs]


def where(box, w, h) -> str:
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    if w >= h:
        return "слева" if cx < 0.4 else "справа" if cx > 0.6 else "в центре"
    return "сверху" if cy < 0.4 else "снизу" if cy > 0.6 else "в центре"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/info")
def info():
    return {
        "model": META["model"],
        "img": IMG,
        "classes": [pretty(c) for c in CLASSES],
        "val": META.get("val"),
        "trained_on": META.get("trained_on"),
    }


@app.post("/api/predict")
def predict(image: UploadFile = File(...), box: str | None = Form(None)):
    data = image.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Картинка больше 15 МБ")
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Не получилось открыть картинку: нужен JPG, PNG или WebP")

    scale = 1.0
    if max(im.size) > MAX_SIDE:
        scale = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
    w, h = im.size

    if box:
        try:
            x1, y1, x2, y2 = (float(v) * scale for v in box.split(","))
        except ValueError:
            raise HTTPException(400, "box должен быть вида x1,y1,x2,y2")
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        if x2 - x1 < 16 or y2 - y1 < 16:
            raise HTTPException(400, "Рамка слишком маленькая")
        wins = [(round(x1), round(y1), round(x2), round(y2))]
    else:
        wins = windows(w, h)

    batch = []
    for b in wins:
        x = preprocess(im.crop(b))
        batch += [x, x[:, :, ::-1]]  # + horizontal flip (test-time augmentation)
    t0 = time.perf_counter()
    logits = SESSION.run(None, {INPUT: np.ascontiguousarray(np.stack(batch), dtype=np.float32)})[0]
    ms = (time.perf_counter() - t0) * 1000
    probs = softmax(logits).reshape(len(wins), 2, -1).mean(1)

    best = int(probs.max(1).argmax())
    p = probs[best]
    order = np.argsort(-p)[:5]
    main_cls = int(order[0])

    def to_orig(b):
        return [round(v / scale) for v in b]

    also: dict[int, dict] = {}
    for i, b in enumerate(wins):
        c = int(probs[i].argmax())
        conf = float(probs[i, c])
        if c != main_cls and conf >= ALSO_MIN and conf > also.get(c, {}).get("prob", 0):
            also[c] = {"name": pretty(CLASSES[c]), "prob": conf, "box": to_orig(b), "where": where(b, w, h)}

    return {
        "top": [{"name": pretty(CLASSES[i]), "id": CLASSES[i], "prob": float(p[i])} for i in order],
        "confident": bool(p[main_cls] >= CONFIDENT),
        "box": to_orig(wins[best]),
        "where": where(wins[best], w, h) if len(wins) > 1 else None,
        "mode": "box" if box else "auto",
        "views": len(wins),
        "also": sorted(also.values(), key=lambda a: -a["prob"]),
        "ms": round(ms),
    }
