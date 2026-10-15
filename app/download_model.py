"""Download the trained model (model.onnx + meta.json) into app/model/ from the project's GitHub Releases.

    python download_model.py                              # repo detected from `git remote origin`
    python download_model.py --repo user/simpsons-who     # or name it explicitly
    python download_model.py --repo user/simpsons-who --tag v1.0

Standard library only: works before `pip install`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["meta.json", "model.onnx"]


def repo_from_git() -> str | None:
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True,
                             text=True, cwd=HERE).stdout.strip()
    except (FileNotFoundError, OSError):
        return None
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else None


def download(url: str, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dst.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print(f"\r  {dst.name}: {done / 1e6:.1f} MB ✓" + " " * 20)
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="GitHub user/repo (default: from git remote origin)")
    ap.add_argument("--tag", default="latest", help="release tag, e.g. v1.0 (default: latest release)")
    ap.add_argument("--base-url", help="download from this URL prefix instead of GitHub Releases")
    args = ap.parse_args()

    if args.base_url:
        base = args.base_url.rstrip("/")
    else:
        repo = args.repo or repo_from_git()
        if not repo:
            sys.exit("Не нашёл репозиторий: запусти внутри git clone или укажи --repo user/simpsons-who")
        base = (f"https://github.com/{repo}/releases/latest/download" if args.tag == "latest"
                else f"https://github.com/{repo}/releases/download/{args.tag}")

    out = HERE / "model"
    out.mkdir(exist_ok=True)
    print(f"Скачиваю модель из {base}")
    for name in FILES:
        try:
            download(f"{base}/{name}", out / name)
        except urllib.error.HTTPError as e:
            sys.exit(f"✗ {name}: HTTP {e.code}. Есть ли в релизе файлы {', '.join(FILES)}?")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    print(f"Готово: {meta['model']}, {len(meta['classes'])} персонажей → {out}")


if __name__ == "__main__":
    main()
