"""
Offline inference evaluation for Disaster-Damage-AI (development tool).

IMPORTANT HONESTY NOTE
-----------------------
There is NO ground-truth segmentation for the bundled test images, so this
script does NOT (and cannot) compute accuracy / IoU / Dice. It only measures
things that are actually measurable without labels:

  * per-class pixel distribution
  * softmax confidence statistics (mean / median / p10 / low-confidence share)
  * number of connected predicted regions and tiny (<= 3 px) regions per class
  * agreement between predictions on the original input vs. flipped input
    (horizontal / vertical) -- a proxy for prediction stability
  * effect of horizontal-flip test-time augmentation (TTA)
  * effect of a conservative 3x3 majority speck filter
  * wall-clock inference time

Variants compared (identical model and weights in all cases):
  base      : exact replication of the current backend/main.py pipeline
              (EXIF transpose -> RGB -> resize 512x512 BICUBIC -> /255,
               channels = [before RGB, after RGB])
  imagenet  : same, but ImageNet mean/std normalization instead of /255
  bilinear  : same as base, but BILINEAR resize instead of BICUBIC

Run from the project root (model path is relative to CWD):
    .venv\\Scripts\\python.exe scripts\\evaluate_inference.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # backend/model.py loads "backend/best_model.pth" relative to CWD

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import PIL

from backend.model import model, DEVICE  # noqa: E402  (loads the trained model)

PAIRS = [
    ("pair1", "test_images/before1.jpg", "test_images/after1.jpg"),
    ("pair2", "test_images/before2.jpg", "test_images/after2.jpg"),
]

INPUT_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUT_DIR = ROOT / "scripts" / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = np.array(
    [[46, 204, 113], [241, 196, 15], [230, 126, 34], [231, 76, 60], [142, 68, 173]],
    dtype=np.uint8,
)

results: dict = {"pillow_version": PIL.__version__, "torch_version": torch.__version__, "pairs": {}}


# --------------------------------------------------------------------------- #
# Checkpoint verification
# --------------------------------------------------------------------------- #
def verify_checkpoint() -> None:
    print("== checkpoint verification ==")
    sd = torch.load("backend/best_model.pth", map_location="cpu", weights_only=True)
    w = sd["encoder.patch_embed1.proj.weight"]
    print(f"  state_dict tensors       : {len(sd)}")
    print(f"  first conv weight shape  : {tuple(w.shape)}  (out_ch, in_ch, kH, kW)")
    print(f"  first conv in_channels   : {w.shape[1]}  -> 6 means the checkpoint was "
          f"trained with the 6-channel input")
    results["checkpoint"] = {"n_tensors": len(sd), "first_conv_shape": list(w.shape)}


# --------------------------------------------------------------------------- #
# Preprocessing variants
# --------------------------------------------------------------------------- #
def preprocess(path: str, resample, norm: str) -> torch.Tensor:
    """Single image -> (3, 512, 512) float tensor."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if norm == "imagenet":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def make_input(before: str, after: str, resample, norm: str) -> torch.Tensor:
    """Image pair -> (1, 6, 512, 512) tensor, channels = [before RGB, after RGB]."""
    return (
        torch.cat([preprocess(before, resample, norm), preprocess(after, resample, norm)], dim=0)
        .unsqueeze(0)
        .to(DEVICE)
    )


def forward_probs(x: torch.Tensor) -> torch.Tensor:
    """Run the model, return softmax probabilities (1, 5, H, W)."""
    with torch.inference_mode():
        logits = model(x)
    if tuple(logits.shape) != (1, 5, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"unexpected model output shape {tuple(logits.shape)}")
    return F.softmax(logits, dim=1)


# --------------------------------------------------------------------------- #
# Metrics (no ground truth involved)
# --------------------------------------------------------------------------- #
def pred_from(probs: torch.Tensor) -> np.ndarray:
    return probs.argmax(dim=1)[0].cpu().numpy()


def conf_stats(probs: torch.Tensor) -> dict:
    a = probs.max(dim=1)[0].flatten().cpu().numpy()
    return {
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "low_conf_share": round(float((a < 0.5).mean()), 6),
    }


def distribution(pred: np.ndarray) -> dict:
    return {
        f"class_{c}": round(100.0 * float((pred == c).sum()) / pred.size, 2) for c in range(5)
    }


def label_components(mask: np.ndarray):
    """Two-pass union-find labeling with 8-connectivity.
    Returns a label map cropped to the mask bounding box (labels >= 1),
    or None when the mask is empty. Label ids are not compact."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    sub = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = sub.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    nxt = 1
    for y in range(h):
        row = sub[y]
        lab_row = labels[y]
        lab_up = labels[y - 1] if y > 0 else None
        for x in range(w):
            if not row[x]:
                continue
            cands = []
            if x > 0 and lab_row[x - 1]:
                cands.append(lab_row[x - 1])
            if lab_up is not None:
                for xx in (x - 1, x, x + 1):
                    if 0 <= xx < w and lab_up[xx]:
                        cands.append(lab_up[xx])
            if not cands:
                labels[y, x] = nxt
                parent.append(nxt)
                nxt += 1
            else:
                roots = [find(c) for c in cands]
                r = min(roots)
                for q in roots:
                    if q != r:
                        parent[q] = r
                labels[y, x] = r

    lut = np.zeros(nxt, dtype=np.int32)
    for i in range(1, nxt):
        lut[i] = find(i)
    return lut[labels]


def region_stats(pred: np.ndarray) -> dict:
    """Per class: number of 8-connected regions, tiny regions (<= 3 px),
    and pixels living in tiny regions."""
    out = {}
    for c in range(5):
        lab = label_components(pred == c)
        if lab is None:
            out[c] = {"regions": 0, "tiny_regions": 0, "tiny_pixels": 0}
            continue
        sizes = np.bincount(lab.ravel())
        sizes = sizes[sizes > 0]
        out[c] = {
            "regions": int(sizes.size),
            "tiny_regions": int((sizes <= 3).sum()),
            "tiny_pixels": int(sizes[sizes <= 3].sum()),
        }
    return out


def majority_speck_filter(pred: np.ndarray, max_own: int = 3):
    """Conservative 3x3 neighborhood majority vote.

    A pixel is rewritten ONLY when its own class appears at most `max_own`
    times in its 3x3 neighborhood (including itself) while some other class is
    strictly more frequent there. With max_own=3 this removes isolated
    1-3 pixel specks; any 2x2 block (own count = 4) survives.
    Returns (filtered_pred, n_changed_pixels).
    """
    h, w = pred.shape
    flat = pred.ravel()
    counts = np.zeros((h * w, 5), dtype=np.int16)
    pad = np.pad(pred, 1, mode="edge")
    rows = np.arange(h * w)
    for dy in range(3):
        for dx in range(3):
            win = pad[dy: dy + h, dx: dx + w].ravel()
            counts[rows, win] += 1
    own = counts[rows, flat]
    best = counts.argmax(axis=1)
    best_count = counts.max(axis=1)
    change = (own <= max_own) & (best_count > own)
    out = flat.copy()
    out[change] = best[change]
    return out.reshape(h, w), int(change.sum())


def save_overlay(name: str, after_path: str, pred: np.ndarray, alpha: float = 0.45) -> None:
    img = (
        ImageOps.exif_transpose(Image.open(after_path)).convert("RGB").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.BICUBIC
        )
    )
    base = np.asarray(img, np.float32)
    color = CLASS_COLORS[pred].astype(np.float32)
    out = (1 - alpha) * base + alpha * color
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(OUT_DIR / f"{name}.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def fmt_regions(rs: dict) -> str:
    parts = []
    for c in range(5):
        r = rs[c]
        parts.append(f"c{c}: {r['regions']} regions / {r['tiny_regions']} tiny(<=3px)")
    return "; ".join(parts)


def main() -> None:
    verify_checkpoint()
    print(f"\ndevice: {DEVICE}\n")

    for pair_name, before_path, after_path in PAIRS:
        print(f"===================== {pair_name}: {before_path} + {after_path} =====================")
        pair_res: dict = {}

        # ---- inputs -------------------------------------------------------- #
        x_base = make_input(before_path, after_path, Image.BICUBIC, "01")  # current pipeline
        x_imn = make_input(before_path, after_path, Image.BICUBIC, "imagenet")
        x_bil = make_input(before_path, after_path, Image.BILINEAR, "01")

        # ---- forwards ------------------------------------------------------ #
        t0 = time.perf_counter()
        probs_base = forward_probs(x_base)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs_h = torch.flip(forward_probs(torch.flip(x_base, dims=[3])), dims=[3])
        t_h = time.perf_counter() - t0  # hflip forward (same cost as base)

        probs_v = torch.flip(forward_probs(torch.flip(x_base, dims=[2])), dims=[2])

        probs_imn = forward_probs(x_imn)
        probs_imn_h = torch.flip(forward_probs(torch.flip(x_imn, dims=[3])), dims=[3])
        probs_bil = forward_probs(x_bil)

        print(f"  forward time: {t_base:.2f} s (per pass; hflip pass: {t_h:.2f} s)")

        pred_base = pred_from(probs_base)
        pred_h = pred_from(probs_h)
        pred_v = pred_from(probs_v)
        pred_imn = pred_from(probs_imn)
        pred_imn_h = pred_from(probs_imn_h)
        pred_bil = pred_from(probs_bil)

        # ---- stability (flip agreement) ------------------------------------- #
        h_agree = 100.0 * float((pred_base == pred_h).mean())
        v_agree = 100.0 * float((pred_base == pred_v).mean())
        h_agree_imn = 100.0 * float((pred_imn == pred_imn_h).mean())

        # ---- TTA (average of original + hflip probabilities) --------------- #
        probs_tta = 0.5 * (probs_base + probs_h)
        pred_tta = pred_from(probs_tta)
        tta_changed = 100.0 * float((pred_base != pred_tta).mean())

        # ---- conservative speck filter -------------------------------------- #
        pred_filt, n_changed = majority_speck_filter(pred_base)
        pred_tta_filt, n_changed_tta = majority_speck_filter(pred_tta)

        # ---- report ---------------------------------------------------------- #
        for variant, pred, probs in (
            ("base", pred_base, probs_base),
            ("imagenet", pred_imn, probs_imn),
            ("bilinear", pred_bil, probs_bil),
            ("tta_h", pred_tta, probs_tta),
            ("base_filtered", pred_filt, probs_base),
            ("tta_h_filtered", pred_tta_filt, probs_tta),
        ):
            dist = distribution(pred)
            conf = conf_stats(probs)
            regs = region_stats(pred)
            print(f"\n  -- {variant} --")
            print(f"     distribution : {dist}")
            print(f"     confidence   : mean={conf['mean']:.4f} median={conf['median']:.4f} "
                  f"p10={conf['p10']:.4f} low(<0.5)={conf['low_conf_share']*100:.2f}%")
            print(f"     regions      : {fmt_regions(regs)}")
            pair_res[variant] = {
                "distribution": dist,
                "confidence": conf,
                "regions": regs,
                "overlay": None,
            }

        print(f"\n  flip agreement (base)      : h={h_agree:.2f}%  v={v_agree:.2f}%")
        print(f"  flip agreement (imagenet)  : h={h_agree_imn:.2f}%")
        print(f"  TTA changed vs base        : {tta_changed:.2f}% of pixels")
        print(f"  speck filter changed       : base={n_changed}px  tta={n_changed_tta}px")
        print(f"  distribution shift by filter: {distribution(pred_filt)}")

        save_overlay(f"{pair_name}_base", after_path, pred_base)
        save_overlay(f"{pair_name}_imagenet", after_path, pred_imn)
        save_overlay(f"{pair_name}_tta_filtered", after_path, pred_tta_filt)

        pair_res["agreement"] = {"hflip": h_agree, "vflip": v_agree, "imagenet_hflip": h_agree_imn}
        pair_res["tta_changed_pct"] = tta_changed
        pair_res["speck_filter_changed_px"] = {"base": n_changed, "tta": n_changed_tta}
        pair_res["forward_time_s"] = t_base
        results["pairs"][pair_name] = pair_res
        print()

    import json

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {OUT_DIR / 'results.json'}")
    print(f"saved overlays: {OUT_DIR / pair_name}_*.png")


if __name__ == "__main__":
    main()
