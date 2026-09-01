# Project Status — Disaster Damage AI

> Last updated: September 2026
> Hackathon: Alibaba Cloud AI Hackathon Pakistan 2026
> Status: **Hackathon Demo Ready**

---

## Hackathon Demo Readiness

The project is fully ready for a 60–90 second hackathon demonstration. A judge can test the complete workflow within one minute:

1. Start the backend: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
2. Open `frontend/index.html` in a browser
3. Upload `test_images/before1.jpg` (Before) and `test_images/after1.jpg` (After)
4. Click **Analyze Damage**
5. Review: Overall Assessment → Class Distribution → Damage Overlay (try the opacity slider) → Before vs After
6. Click **Download Analysis Report** to save an HTML report

No cloud deployment or external APIs are required.

---

## 60–90 Second Demo Flow (Live Presentation)

**Use Test Case 2 (`before2.jpg` → `after2.jpg`)** — it produces the most visually meaningful result (MODERATE DAMAGE, 15.0% affected vs 85.0% unaffected, 88% softmax confidence, clear overlay contrast). Full presenter script, timings, backup plans and "do not click" list: **`DEMO_GUIDE.md`**.

| Time | Action | What to show/say |
|---|---|---|
| 0:00–0:10 | App open, no clicks | Problem/solution pitch (hero + feature cards) |
| 0:10–0:17 | Click **Test Case 2** card, dismiss alert | Built-in demo guide |
| 0:17–0:33 | Upload `before2.jpg`, then `after2.jpg` | Live previews |
| 0:33–0:40 | Click **Analyze Damage** | Loading spinner (~2.5 s CPU inference) |
| 0:40–0:55 | Scroll to results | MODERATE DAMAGE badge, 15.0% affected / 85.0% unaffected, 88% softmax confidence |
| 0:55–1:03 | Point at class bars | 5-class damage distribution |
| 1:03–1:17 | Damage Overlay, drag slider 60 → 80 | Interactive damage map + legend |
| 1:17–1:26 | Before vs After, then Download button | Shareable HTML report |
| 1:26–1:30 | Scroll to footer | AI-assisted assessment disclaimer |

**Golden rules:** verify `/health` shows `model_loaded: true` before starting; never refresh mid-demo; never re-click Analyze while loading.

---

## Test Results (September 2026 Audit)

> Demo-preparation re-verification (September 2026): Test Case 2 was re-tested end-to-end
> through the live backend API (`scripts/demo_check.py`) and through the browser UI
> (12/12 checkpoints passed, zero console errors). Backend API + frontend E2E tests below
> remain valid; no application code was changed.

### Backend API Tests — All Passed

| Test | Result |
|---|---|
| `GET /` | 200 — API online |
| `GET /health` | 200 — model loaded, 5 classes, CPU device |
| `POST /predict` — Test Case 1 | 200 — prediction succeeds |
| `POST /predict` — Test Case 2 | 200 — prediction succeeds |
| Empty file handling | 400 — "One or both uploaded files are empty." |
| Invalid image handling | 400 — "One or both files are not valid images." |
| Oversized file handling | 400 — "The 'before' image is too large (11.0 MB). Maximum allowed size is 10 MB." |
| Identical file warning | 200 — warning returned in `warnings` array |

### Frontend E2E Tests — All 14 Features Passed

| # | Feature | Status |
|---|---------|--------|
| 1 | Initial page (title, feature cards, how-it-works, demo guide) | PASS |
| 2 | Error handling (no images) | PASS |
| 3 | Image upload + preview | PASS |
| 4 | Analyze button + loading state | PASS |
| 5 | Overall Assessment (badge, stats, explanation) | PASS |
| 6 | Class Distribution (5 classes, percentages, pixels) | PASS |
| 7 | Damage Overlay (canvas, legend) | PASS |
| 8 | Opacity Slider (changes overlay at 20% and 80%) | PASS |
| 9 | Before vs After comparison | PASS |
| 10 | Download Report button | PASS |
| 11 | AI-assisted disclaimer + Footer | PASS |
| 12 | Test Case 2 results | PASS |
| 13 | No JavaScript console errors | PASS |
| 14 | No broken elements or missing UI | PASS |

### Test Case 1 — Actual Model Output

- **Images:** `before1.jpg` → `after1.jpg`
- **Assessment:** SEVERE DAMAGE
- **Affected Area:** 43.7%
- **Unaffected Area:** 56.3%
- **Class Distribution:**
  - No Damage: 56.31% (147,613 px)
  - Minor Damage: 35.25% (92,399 px)
  - Moderate Damage: 0.00% (0 px)
  - Severe Damage: 0.48% (1,256 px)
  - Complete Destruction: 7.96% (20,876 px)
- **Sum:** 100.00% (262,144 pixels = 512 × 512)
- **Prediction Quality:** 77.6% (softmax confidence)
- **Flip View Agreement:** 0.824
- **Inference Time:** 2.60s (CPU)

### Test Case 2 — Actual Model Output

- **Images:** `before2.jpg` → `after2.jpg`
- **Assessment:** MODERATE DAMAGE
- **Affected Area:** 15.0%
- **Unaffected Area:** 85.0%
- **Class Distribution:**
  - No Damage: 85.03% (222,906 px)
  - Minor Damage: 6.15% (16,132 px)
  - Moderate Damage: 0.00% (13 px)
  - Severe Damage: 0.00% (0 px)
  - Complete Destruction: 8.81% (23,093 px)
- **Sum:** 99.99% (262,144 pixels = 512 × 512)
- **Prediction Quality:** 88.0% (softmax confidence)
- **Flip View Agreement:** 0.9148
- **Inference Time:** 2.33s (CPU)

### Performance

| Metric | Value |
|---|---|
| Model loading time | ~5 seconds (first startup) |
| Test Case 1 inference | 2.60 seconds |
| Test Case 2 inference | 2.33 seconds |
| Device | CPU (auto-detects CUDA if available) |

CPU inference is acceptable for a hackathon demo — results appear within 3 seconds.

---

## Current Working Features

### Backend
- FastAPI server running on `http://127.0.0.1:8000`
- `GET /` — API health check
- `GET /health` — Device, model status, class count, and Python version
- `POST /predict` — Image analysis endpoint (returns prediction quality score, confidence diagnostics, and input warnings)
- U-Net segmentation model with mit_b3 encoder
- 6-channel input (before RGB + after RGB)
- 5-class pixel-level damage classification
- ImageNet per-channel normalization (checkpoint fine-tuned from ImageNet-pretrained encoder; verified with `scripts/evaluate_inference.py`)
- Horizontal-flip test-time augmentation (two views averaged)
- Conservative 3×3 majority speck filter (removes isolated 1–3 pixel regions; measured effect <0.1 percentage points on class shares)
- Base64 RGBA damage mask generation
- Input diagnostics: identical before/after files and mismatched aspect ratios produce warnings
- CPU inference (auto-detects CUDA if available)
- CORS middleware for cross-origin request support
- File size validation (10 MB limit per upload)
- EXIF orientation correction for rotated photos
- Input validation: empty files, invalid images, unsupported formats

### Frontend
- Before/after image upload with live previews
- File type validation (JPG, JPEG, PNG, WEBP) and file size check
- Loading spinner during analysis
- Overall assessment card with:
  - Damage level badge (Low / Moderate / Severe)
  - Total affected area percentage
  - Unaffected area percentage
  - Dominant damage class
  - Damage types detected count
  - Prediction quality (softmax confidence) with visual bar
  - Plain-English explanation
- Class distribution bars with percentages and pixel counts
- Damage overlay with adjustable opacity slider (0–100%)
- Color legend (5 classes)
- Before vs After side-by-side comparison
- Downloadable HTML analysis report (includes images, stats, overlay, AI disclaimer)
- Quick Demo Guide with test case cards
- "How It Works" 6-step visual guide
- Feature highlight cards
- AI-assisted assessment disclaimer (displayed in UI and in downloaded report)
- Professional error messages with API detail extraction
- Responsive layout (desktop, laptop, mobile)
- "Analysis Complete" status badge

### Test Images
- `test_images/before1.jpg` — Sample before-disaster aerial photo
- `test_images/after1.jpg` — Sample after-disaster aerial photo
- `test_images/before2.jpg` — Additional before image
- `test_images/after2.jpg` — Additional after image

---

## Current Model / Inference Pipeline

| Property | Value |
|---|---|
| Framework | PyTorch |
| Architecture | U-Net (`segmentation_models_pytorch`) |
| Encoder | `mit_b3` (Mix Transformer B3) |
| Input | 6 channels (3 RGB before + 3 RGB after) |
| Output | 5 damage classes |
| Input resolution | 512 × 512 pixels (resized at inference) |
| Normalization | ImageNet per-channel mean/std |
| Test-time augmentation | Horizontal flip (two views, probability maps averaged) |
| Post-processing | Conservative 3×3 majority speck filter |
| Trained weights | `backend/best_model.pth` (~190 MB) |
| Inference device | CPU (auto-detects CUDA if available) |

---

## Known Limitations

1. **Class labels are unverified** — The mapping of class_0–class_4 to human-readable names exists only in the frontend HTML. No training code, dataset metadata, or documentation is included in this repository to confirm these labels.

2. **CPU-only inference** — The model runs on CPU (~2.3–2.6 seconds per prediction with flip test-time augmentation). CUDA auto-detection is present but not actively configured.

3. **No cloud deployment** — The application currently runs only locally. No deployment to Alibaba Cloud or any other platform has been done yet.

4. **No real-world area estimation** — Percentages represent image pixels, not geographic area.

5. **Fixed resolution** — Images are resized to 512×512 before processing.

6. **No batch processing** — Only one image pair can be analyzed at a time.

7. **No scientific evaluation** — No accuracy, IoU, Dice, or other metrics have been measured against ground-truth masks. See the Research & Evaluation section in README.md.

8. **Appearance sensitivity on identical pairs** — If identical files are uploaded as both before and after, the model can still report damage classes based on image appearance; the API returns a warning in that case.

---

## Future Research Work

### For Cloud Deployment
1. **Alibaba Cloud deployment** — Deploy the backend on ECS, PAI, or Function Compute
2. **GPU inference** — Enable CUDA acceleration for faster predictions
3. **HTTPS and authentication** — Secure the API for public access

### For Scientific Evaluation
4. **Verified training metadata** — Add training scripts, dataset documentation, and validated class definitions
5. **Ground-truth evaluation** — Measure IoU, Dice coefficient, precision, recall on annotated test sets
6. **Baseline comparison** — Compare against simpler approaches (image differencing, threshold-based methods)
7. **Real-world area estimation** — Integrate geospatial data to approximate square meters

### Lower Priority
8. **Multi-language support** — Localize the frontend
9. **Batch processing** — Allow multiple image pair uploads
10. **User accounts** — Save analysis history
11. **Uncertainty estimation** — Monte Carlo dropout or ensemble methods for per-pixel confidence intervals

---

## Important Commands

### Start the backend

```powershell
cd Disaster-Damage-AI
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Open the frontend

Open `frontend/index.html` in a web browser. The backend must be running on port 8000.

### Install dependencies

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Test with sample images

1. Open `frontend/index.html`
2. Upload `test_images/before1.jpg` and `test_images/after1.jpg`
3. Click **Analyze Damage**
4. Wait ~3 seconds for CPU inference

### Check API health

Open `http://127.0.0.1:8000/health` in your browser or run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

## File Summary

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI server, `/predict` endpoint, preprocessing, validation |
| `backend/model.py` | U-Net model definition, weight loading |
| `backend/best_model.pth` | Trained model weights (~190 MB) |
| `frontend/index.html` | Complete web application (HTML + CSS + JS) |
| `test_images/` | Sample before/after disaster images |
| `scripts/` | Offline development/evaluation tools (not used by the API) |
| `requirements.txt` | Python package dependencies |
| `README.md` | Full project documentation |
| `DEMO_GUIDE.md` | 60–90 second hackathon demo script, timings and backup plans |
| `PROJECT_STATUS.md` | This file |
| `.gitignore` | Git exclusion rules |
