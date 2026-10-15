"""Export a trained checkpoint to ONNX for the web service (app/), and check it gives the same answers.

    python classification/export_onnx.py --ckpt runs_cls/convnext_s_full/best.pt --out app/model

Writes app/model/model.onnx (single file, dynamic batch) and app/model/meta.json
(image size, normalization, class names, validation metrics).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import load_checkpoint  # noqa: I001

import numpy as np  # noqa: E402
import torch  # noqa: E402


def export(model, img: int, path: Path) -> None:
    x = torch.randn(2, 3, img, img)
    kw = dict(input_names=["input"], output_names=["logits"])
    try:
        # Classic exporter: one self-contained file, no extra dependencies
        torch.onnx.export(model, (x,), str(path), opset_version=17, dynamo=False,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **kw)
    except Exception as e:  # newer PyTorch without the classic exporter
        print(f"classic exporter unavailable ({type(e).__name__}), using dynamo exporter")
        import onnx

        batch = torch.export.Dim("batch", min=1, max=64)
        torch.onnx.export(model, (x,), str(path), dynamo=True, dynamic_shapes=({0: batch},), **kw)
        m = onnx.load(str(path), load_external_data=True)  # fold weights back into one file
        data = path.with_name(path.name + ".data")
        onnx.save(m, str(path), save_as_external_data=False)
        if data.exists():
            data.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("app/model"))
    args = ap.parse_args()

    model, meta = load_checkpoint(args.ckpt, torch.device("cpu"))
    model = model.float().eval().to(memory_format=torch.contiguous_format)
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "model.onnx"

    t0 = time.time()
    export(model, meta["img"], onnx_path)
    print(f"exported in {time.time() - t0:.0f}s → {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB)")

    # Same input through PyTorch and ONNX Runtime must give the same logits
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, meta["img"], meta["img"])
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    same_top1 = bool((ref.argmax(1) == got.argmax(1)).all())
    print(f"PyTorch vs ONNX Runtime: max |Δlogit| = {diff:.2e}, same top-1: {same_top1}")
    if diff > 1e-2 or not same_top1:
        raise SystemExit("✗ ONNX output differs from PyTorch — do not deploy this file")

    info = {
        "model": meta["model"],
        "img": meta["img"],
        "mean": meta["mean"],
        "std": meta["std"],
        "classes": meta["classes"],
        "val": meta.get("val"),
        "trained_on": "100% of data" if meta.get("val") is None else "90% (10% validation)",
        "source": str(args.ckpt),
        "exported": time.strftime("%Y-%m-%d %H:%M"),
    }
    (args.out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"✓ {args.out / 'meta.json'}: {len(info['classes'])} classes, {info['img']}px")


if __name__ == "__main__":
    main()
