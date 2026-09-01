"""Validate saved /predict API responses (development tool)."""
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts" / "eval_output"

ok = True
for name in ("api_pair1", "api_pair2"):
    with open(OUT / f"{name}.json") as f:
        r = json.load(f)

    checks = []

    # Required frontend-contract fields
    for field in ("success", "dominant_class", "classes", "damage_mask", "prediction_quality"):
        checks.append((field in r, f"missing field '{field}'"))
    checks.append((r.get("success") is True, "success is not True"))

    # All 5 classes present with pixels + percentage
    classes = r.get("classes", {})
    for c in range(5):
        key = f"class_{c}"
        checks.append((key in classes, f"missing class '{key}'"))
        if key in classes:
            checks.append(("pixels" in classes[key] and "percentage" in classes[key],
                           f"'{key}' missing pixels/percentage"))

    total_pct = sum(v["percentage"] for v in classes.values())
    checks.append((abs(total_pct - 100.0) < 0.05, f"percentages sum to {total_pct}"))

    total_px = sum(v["pixels"] for v in classes.values())
    checks.append((total_px == 512 * 512, f"pixels sum to {total_px} (expected 262144)"))

    checks.append((isinstance(r.get("dominant_class"), int) and 0 <= r["dominant_class"] <= 4,
                   "dominant_class invalid"))
    checks.append((isinstance(r.get("prediction_quality"), (int, float))
                   and 0 <= r["prediction_quality"] <= 100,
                   "prediction_quality out of range"))

    # Mask: valid base64 RGBA PNG at model resolution
    mask = Image.open(io.BytesIO(base64.b64decode(r["damage_mask"])))
    checks.append((mask.size == (512, 512), f"mask size {mask.size}"))
    checks.append((mask.mode == "RGBA", f"mask mode {mask.mode}"))

    # New additive fields
    cd = r.get("confidence_details", {})
    checks.append(("confidence_details" in r, "missing confidence_details"))
    for k in ("metric", "mean", "median", "p10", "low_confidence_pixel_share", "flip_view_agreement"):
        checks.append((k in cd, f"confidence_details missing '{k}'"))
    checks.append((isinstance(r.get("warnings"), list), "warnings is not a list"))

    failed = [msg for passed, msg in checks if not passed]
    status = "OK" if not failed else f"FAILED: {failed}"
    if failed:
        ok = False

    print(f"{name}: {status}")
    print(f"  dominant_class      : {r['dominant_class']}")
    print(f"  prediction_quality  : {r['prediction_quality']}")
    print(f"  classes             : " + ", ".join(
        f"c{c}={classes[f'class_{c}']['percentage']}%" for c in range(5)))
    print(f"  confidence_details  : {cd}")
    print(f"  warnings            : {r.get('warnings')}")
    print(f"  mask                : {mask.size} {mask.mode}")
    print()

sys.exit(0 if ok else 1)
