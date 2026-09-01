"""
Normalization diagnostics for the trained checkpoint (development tool).

Goal: determine which input normalization the checkpoint in backend/best_model.pth
was trained with, using evidence that does NOT require ground truth.

Tests:
  1. Scale of the first-layer (patch_embed1.proj) weights.
     - ImageNet-normalized training keeps first-conv weights at pretrained
       magnitude (std roughly 0.02-0.05 for a 7x7 conv).
     - [0,1] inputs are ~4x smaller in std than ImageNet-normalized inputs,
       so a [0,1]-trained first conv is ~4x LARGER; [0,255]-trained ~255x larger.
  2. Identical-pair sanity check (before == after). A change/damage model shown
     the SAME image in both halves should predict ~100% "no change/no damage"
     when fed the normalization it was trained with. Fabricating large damage
     on an identical pair indicates the wrong normalization.
  3. Behavior of other candidate scalings ([0,1], imagenet, [-1,1], [0,255])
     on the real test pair for reference.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\diagnose_normalization.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import evaluate_inference as ev  # reuse model loading + preprocessing helpers

IMAGENET_MEAN = ev.IMAGENET_MEAN
IMAGENET_STD = ev.IMAGENET_STD


def preprocess_scaled(path: str, scaling: str) -> torch.Tensor:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = img.resize((512, 512), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if scaling == "01":
        pass
    elif scaling == "imagenet":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    elif scaling == "pm1":
        arr = arr * 2.0 - 1.0
    elif scaling == "0255":
        arr = arr * 255.0
    else:
        raise ValueError(scaling)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def run_pair(before: str, after: str, scaling: str) -> dict:
    x = (
        torch.cat([preprocess_scaled(before, scaling), preprocess_scaled(after, scaling)], dim=0)
        .unsqueeze(0)
        .to(ev.DEVICE)
    )
    probs = ev.forward_probs(x)
    pred = ev.pred_from(probs)
    conf = ev.conf_stats(probs)
    return {"distribution": ev.distribution(pred), "confidence": conf}


def main() -> None:
    print("== 1. first-layer weight scale ==")
    sd = torch.load("backend/best_model.pth", map_location="cpu", weights_only=True)
    w = sd["encoder.patch_embed1.proj.weight"].numpy()  # (64, 6, 7, 7)
    b = sd["encoder.patch_embed1.proj.bias"].numpy()
    print(f"  patch_embed1.proj.weight std : {w.std():.5f}   (mean {w.mean():+.5f})")
    print(f"  before half (ch 0-2) std    : {w[:, :3].std():.5f}")
    print(f"  after  half (ch 3-5) std    : {w[:, 3:].std():.5f}")
    print(f"  patch_embed1.proj.bias  std  : {b.std():.5f}   (mean {b.mean():+.5f})")
    for k in (
        "encoder.patch_embed2.proj.weight",
        "encoder.patch_embed3.proj.weight",
    ):
        if k in sd:
            print(f"  {k} std : {sd[k].numpy().std():.5f}")
    print("  reference: ImageNet-pretrained 7x7 first convs typically have std ~0.02-0.05")

    print("\n== 2. identical-pair sanity check (before == after == after1) ==")
    for scaling in ("01", "imagenet", "pm1", "0255"):
        r = run_pair("test_images/after1.jpg", "test_images/after1.jpg", scaling)
        c0 = r["distribution"]["class_0"]
        print(
            f"  {scaling:>8}: class_0 = {c0:6.2f}%   mean conf = {r['confidence']['mean']:.4f}   "
            f"dist = {r['distribution']}"
        )

    print("\n== 3. identical-pair sanity check (before == after == before2) ==")
    for scaling in ("01", "imagenet", "pm1", "0255"):
        r = run_pair("test_images/before2.jpg", "test_images/before2.jpg", scaling)
        c0 = r["distribution"]["class_0"]
        print(
            f"  {scaling:>8}: class_0 = {c0:6.2f}%   mean conf = {r['confidence']['mean']:.4f}   "
            f"dist = {r['distribution']}"
        )

    print("\n== 4. real pair under all scalings (reference) ==")
    for scaling in ("01", "imagenet", "pm1", "0255"):
        r = run_pair("test_images/before1.jpg", "test_images/after1.jpg", scaling)
        print(
            f"  {scaling:>8}: mean conf = {r['confidence']['mean']:.4f}   dist = {r['distribution']}"
        )


if __name__ == "__main__":
    main()
