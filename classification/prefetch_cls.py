"""Download pretrained backbones into the project (.hf_cache) on a machine with internet.
GPU jobs then run with HF_HUB_OFFLINE=1.

    python classification/prefetch_cls.py                         # default ConvNeXt-Small
    python classification/prefetch_cls.py convnext_base.fb_in22k_ft_in1k_384 eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
"""

from __future__ import annotations

import os
import sys

from common import ROOT  # noqa: F401  (sets HF_HOME inside the project)

import timm  # noqa: E402

DEFAULT = ["convnext_small.fb_in22k_ft_in1k_384"]

for name in sys.argv[1:] or DEFAULT:
    m = timm.create_model(name, pretrained=True, num_classes=0)
    print(f"✓ {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params")
print(f"\nCached in {os.environ['HF_HOME']}")
