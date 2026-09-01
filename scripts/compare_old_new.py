"""
Old-vs-new inference comparison for Disaster-Damage-AI (development tool).

Compares, on the bundled test pairs:
  OLD pipeline : exact replication of backend/main.py before the 2026-09 update
                 ([0,1] scaling, no TTA, no speck filter, torch.no_grad)
  NEW pipeline : the actual functions of the current backend/main.py
                 (ImageNet normalization, horizontal-flip TTA, conservative
                 speck filter, torch.inference_mode)

Measured (all label-free):
  * per-class distribution
  * prediction quality (mean max softmax confidence, same formula in both)
  * isolated tiny (<=3 px) region counts
  * wall-clock inference time
  * response-shape invariants (5 classes present, percentages sum to ~100)

There is no ground truth for these images, so no accuracy/IoU/Dice is computed.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\compare_old_new.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from backend.model import model, DEVICE  # noqa: F401  (loads the trained model)
from backend import main as new_main  # current pipeline under test

import evaluate_inference as ev  # region_stats / distribution helpers

OUT_DIR = ROOT / "scripts" / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("pair1", "test_images/before1.jpg", "test_images/after1.jpg"),
    ("pair2", "test_images/before2.jpg", "test_images/after2.jpg"),
]


# ----------------------------- OLD pipeline --------------------------------- #
def old_preprocess(image: Image.Image) -> torch.Tensor:
    """Exact replication of the pre-update preprocess_image."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((512, 512))  # default resample = BICUBIC for RGB
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.tensor(image_array, dtype=torch.float32)


def old_inference(before_img: Image.Image, after_img: Image.Image):
    combined = (
        torch.cat([old_preprocess(before_img), old_preprocess(after_img)], dim=0)
        .unsqueeze(0)
        .to(DEVICE)
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(combined)
    dt = time.perf_counter() - t0
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    quality = float(F.softmax(output, dim=1).max(dim=1)[0].mean()) * 100
    return pred, quality, dt


# ----------------------------- NEW pipeline --------------------------------- #
def new_inference(before_img: Image.Image, after_img: Image.Image):
    combined = (
        torch.cat([new_main.preprocess_image(before_img), new_main.preprocess_image(after_img)], dim=0)
        .unsqueeze(0)
        .to(DEVICE)
    )
    t0 = time.perf_counter()
    probs, view_agreement = new_main.run_inference(combined)
    dt = time.perf_counter() - t0
    pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
    if new_main.SPECK_FILTER:
        pred = new_main.remove_speck_noise(pred)
    quality = float(probs.max(dim=1)[0].mean()) * 100
    return pred, quality, dt, view_agreement


def tiny_region_summary(pred: np.ndarray) -> str:
    regs = ev.region_stats(pred)
    tiny = sum(regs[c]["tiny_regions"] for c in range(5))
    tiny_px = sum(regs[c]["tiny_pixels"] for c in range(5))
    total = sum(regs[c]["regions"] for c in range(5))
    return f"{total} regions, {tiny} tiny(<=3px), {tiny_px} px in tiny"


def check_invariants(pred: np.ndarray, quality: float) -> list:
    problems = []
    dist = ev.distribution(pred)
    if abs(sum(dist.values()) - 100.0) > 0.05:
        problems.append(f"percentages sum to {sum(dist.values()):.2f}")
    if not (0.0 <= quality <= 100.0):
        problems.append(f"quality out of range: {quality}")
    return problems


def main() -> None:
    for name, before_path, after_path in PAIRS:
        before_img = Image.open(before_path)
        after_img = Image.open(after_path)

        print(f"================= {name} =================")
        pred_old, q_old, t_old = old_inference(before_img, after_img)
        pred_new, q_new, t_new, agree = new_inference(before_img, after_img)

        print(f"  OLD : time {t_old:5.2f}s | quality {q_old:5.1f}% | {ev.distribution(pred_old)}")
        print(f"        tiny regions: {tiny_region_summary(pred_old)}")
        print(f"  NEW : time {t_new:5.2f}s | quality {q_new:5.1f}% | flip-agreement {agree*100:.1f}%")
        print(f"        dist {ev.distribution(pred_new)}")
        print(f"        tiny regions: {tiny_region_summary(pred_new)}")

        for tag, pred in (("old", pred_old), ("new", pred_new)):
            problems = check_invariants(pred, q_old if tag == "old" else q_new)
            print(f"        invariants[{tag}]: {'OK' if not problems else problems}")

        changed = 100.0 * float((pred_old != pred_new).mean())
        print(f"  pixels changed old -> new: {changed:.2f}%")

        ev.save_overlay(f"{name}_compare_old", after_path, pred_old)
        ev.save_overlay(f"{name}_compare_new", after_path, pred_new)
        print()

    print("overlays saved to scripts/eval_output/<pair>_compare_{old,new}.png")


if __name__ == "__main__":
    main()
