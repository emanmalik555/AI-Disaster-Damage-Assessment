# Disaster Damage AI

An AI-powered web application that assesses disaster damage by comparing before-and-after satellite or aerial images. Built for the **Alibaba Cloud AI Hackathon Pakistan 2026**.

The application uses a deep learning segmentation model to perform pixel-level damage classification, producing a color-coded damage map, percentage breakdowns, and a downloadable analysis report.

---

## Problem

After a natural disaster — earthquake, flood, hurricane, or wildfire — one of the first challenges is assessing the extent of damage across affected areas. Manual inspection is slow, dangerous, and expensive. Comparing before-and-after imagery manually is subjective and time-consuming.

Emergency responders, insurance companies, and government agencies need a fast, consistent, and scalable way to quantify damage from aerial or satellite photographs.

---

## Solution

Disaster Damage AI automates this process:

1. The user uploads a **before-disaster** image and an **after-disaster** image of the same area.
2. A trained **U-Net segmentation model** compares both images and classifies every pixel into one of five damage levels.
3. The frontend displays a **color-coded damage overlay**, percentage breakdowns, an overall assessment, and a downloadable HTML report.

This provides a rapid, repeatable, AI-assisted first assessment that can guide further investigation and resource allocation.

---

## Features

- **Before/After Image Analysis** — Upload two images and get instant results
- **5-Class Damage Segmentation** — Pixel-level classification across five damage levels
- **Damage Percentages** — Exact pixel counts and percentages for each class
- **Visual Damage Overlay** — Color-coded mask rendered on top of the after image
- **Overlay Opacity Control** — Adjustable slider (0–100%) to compare overlay intensity
- **Before vs After Comparison** — Side-by-side image display in the results
- **Overall Damage Assessment** — Calculated affected area, dominant damage class, and plain-English explanation
- **Downloadable Analysis Report** — Self-contained HTML report with images, stats, and AI-assisted disclaimer
- **Responsive Web Interface** — Works on desktop, laptop, and mobile
- **Demo Mode** — Instructions for quick testing with included sample images
- **Frontend Validation** — File type checking and user-friendly error messages

---

## AI Model

The segmentation model is verifiable from the existing code:

| Property | Value |
|---|---|
| Framework | PyTorch |
| Architecture | U-Net (via `segmentation_models_pytorch`) |
| Encoder | `mit_b3` (Mix Transformer B3) |
| Input | 6 channels (3 RGB before + 3 RGB after) |
| Output | 5 damage classes |
| Input resolution | 512 × 512 pixels (resized at inference) |
| Normalization | ImageNet per-channel mean/std |
| Test-time augmentation | Horizontal flip (two views, probability maps averaged) |
| Trained weights | `backend/best_model.pth` (~190 MB) |
| Inference device | CPU (auto-detects CUDA if available) |

The encoder's first convolution layer is replaced with a 6-channel version to accept both images simultaneously. The original RGB weights are duplicated into both halves of the new convolution.

**Why ImageNet normalization:** the checkpoint was fine-tuned from an ImageNet-pretrained `mit_b3` encoder (the duplicated first-convolution weights keep pretrained magnitude), and ImageNet-pretrained encoders expect ImageNet statistics. This was verified with the offline evaluation in `scripts/evaluate_inference.py`: with plain [0,1] scaling the model collapsed to a near-constant ~97–100% "no damage" prediction regardless of input content, while ImageNet statistics produce spatially structured predictions that follow the visible damage in the test images. A conservative 3×3 majority pass additionally removes isolated 1–3 pixel specks (measured effect: 0.02–0.03% of pixels, class percentages shift by less than 0.1 points).

---

## Damage Classes

> **Important:** The following class names are the **current frontend interpretation**. The training code and dataset metadata are not included in this repository, so these labels cannot be independently verified from the project files alone.

| Class ID | Frontend Label | Color |
|---|---|---|
| `class_0` | No Damage | 🟢 Green |
| `class_1` | Minor Damage | 🟡 Yellow |
| `class_2` | Moderate Damage | 🟠 Orange |
| `class_3` | Severe Damage | 🔴 Red |
| `class_4` | Complete Destruction | 🟣 Purple |

---

## How It Works

```
User uploads                Backend                     Frontend
─────────────              ───────                     ────────
Before image  ──┐
                 ├──►  Preprocess (resize,     ──►  Display assessment card
After image   ──┘       normalize, combine)          with stats grid
                          │
                          ▼
                     U-Net segmentation
                     (pixel-level, 5 classes)
                          │
                          ▼
                     Count pixels per class
                     Generate RGBA damage mask
                          │
                          ▼
                     Return JSON:
                     • class percentages
                     • dominant class
                     • base64 damage mask
                                                     ──►  Render damage overlay
                                                          on canvas with slider
                                                     ──►  Show Before vs After
                                                     ──►  Enable report download
```

**Step by step:**

1. User uploads a **before-disaster** image
2. User uploads an **after-disaster** image
3. Frontend sends both images to the backend via `POST /predict`
4. Backend preprocesses both images (EXIF-correct, resize to 512×512, ImageNet normalization, combine into a 6-channel tensor) and runs inference twice — once on the input and once on a horizontally flipped copy — averaging the two probability maps
5. The U-Net model performs **pixel-level segmentation**, assigning each pixel to one of 5 classes
6. Backend counts pixels per class, calculates percentages, and generates a color-coded RGBA mask
7. Backend returns the results as JSON (percentages + base64-encoded mask)
8. Frontend renders the assessment card, class distribution bars, damage overlay, and comparison view

---

## Project Structure

```
Disaster-Damage-AI/
├── backend/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI application, /predict endpoint, image preprocessing
│   ├── model.py             # U-Net model definition, weight loading
│   └── best_model.pth       # Trained model weights (~190 MB)
├── frontend/
│   └── index.html           # Single-page web application (HTML + CSS + JS)
├── test_images/
│   ├── before1.jpg          # Sample before-disaster image
│   ├── after1.jpg           # Sample after-disaster image
│   ├── before2.jpg          # Additional sample before image
│   └── after2.jpg           # Additional sample after image
├── scripts/                 # Offline development/evaluation tools (not used by the API)
│   ├── evaluate_inference.py      # Variant comparison (normalization, TTA, speck filter)
│   ├── diagnose_normalization.py  # Checkpoint normalization diagnostics
│   ├── compare_old_new.py         # Old-vs-new pipeline comparison
│   ├── validate_api_response.py   # /predict response contract checks
│   └── demo_check.py             # Pre-flight check: posts Test Case 2 and prints expected numbers
├── data/                    # Reserved for dataset files (currently empty)
├── docs/                    # Reserved for documentation (currently empty)
├── .gitattributes           # Git LFS tracking configuration
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── PROJECT_STATUS.md        # Development status and next steps
├── DEMO_GUIDE.md            # 60–90 second hackathon demo script
└── .venv/                   # Python virtual environment
```

---

## Installation

### Prerequisites

- **Python 3.10+** (recommended)
- **pip** (Python package manager)
- **Git LFS** (required to download the model weights — see below)

### Model Weights (Git LFS)

The trained model weights (`backend/best_model.pth`, ~190 MB) are stored using [Git LFS](https://git-lfs.com/) (Large File Storage). After cloning, make sure Git LFS is installed and pull the actual model file:

```bash
git lfs install
git lfs pull
```

Verify the file was downloaded correctly:

```bash
git lfs ls-files
```

You should see `backend/best_model.pth` listed. If Git LFS is not installed, the file will contain only a small pointer (~130 bytes) instead of the actual weights, and the model will fail to load.

### Setup

1. **Clone or download** this repository.

2. **Create a virtual environment** (recommended):

   ```powershell
   cd Disaster-Damage-AI
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

   This installs PyTorch, FastAPI, the segmentation models library, and other required packages.

   > **Note:** PyTorch is a large download (~2 GB). Installation may take several minutes.

---

## Running the Backend

With the virtual environment activated:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

You should see:

```
Building model...
Model loaded successfully on cpu
INFO:     Started server process
INFO:     Application startup complete.
```

The backend is now running at `http://127.0.0.1:8000`.

> **Tip:** The first startup takes a few seconds while the model loads into memory.

---

## Running the Frontend

The frontend is a single HTML file. Open it directly in your web browser:

```
frontend/index.html
```

You can double-click the file in your file explorer, or use:

```powershell
start frontend\index.html
```

The frontend communicates with the backend at `http://127.0.0.1:8000`. **The backend must be running for analysis to work.**

---

## Testing

The project includes sample disaster images for testing:

| File | Description |
|---|---|
| `test_images/before1.jpg` | Before-disaster aerial photo |
| `test_images/after1.jpg` | After-disaster aerial photo |
| `test_images/before2.jpg` | Additional before image |
| `test_images/after2.jpg` | Additional after image |

### Quick test:

1. Start the backend (see above)
2. Open `frontend/index.html` in your browser
3. Use the **Quick Demo Guide** section to see which test images to use
4. Upload `test_images/before1.jpg` as the **Before** image
5. Upload `test_images/after1.jpg` as the **After** image
6. Click **🔍 Analyze Damage**
7. Wait ~5 seconds (the model runs on CPU)
8. Review the results: assessment card, class distribution, damage overlay, and comparison view
9. Click **📥 Download Analysis Report** to save an HTML report

---

## API

The backend exposes three endpoints:

### `GET /`

Health check. Returns:

```json
{
  "message": "Disaster Damage AI API is running",
  "status": "online"
}
```

### `GET /health`

Returns device and model information:

```json
{
  "status": "healthy",
  "device": "cpu",
  "model_loaded": true,
  "num_classes": 5,
  "python_version": "3.12.0"
}
```

### `POST /predict`

Performs damage analysis on a pair of images.

**Request:** `multipart/form-data` with two fields:

| Field | Type | Description |
|---|---|---|
| `before` | File | Before-disaster image (JPG, PNG, or WEBP) |
| `after` | File | After-disaster image (JPG, PNG, or WEBP) |

**Response:**

```json
{
  "success": true,
  "dominant_class": 0,
  "classes": {
    "class_0": { "pixels": 147596, "percentage": 56.31 },
    "class_1": { "pixels": 92411, "percentage": 35.25 },
    "class_2": { "pixels": 0, "percentage": 0.0 },
    "class_3": { "pixels": 1257, "percentage": 0.48 },
    "class_4": { "pixels": 20880, "percentage": 7.96 }
  },
  "damage_mask": "<base64-encoded RGBA PNG>",
  "prediction_quality": 77.6,
  "confidence_details": {
    "metric": "mean_max_softmax_confidence",
    "mean": 0.7764,
    "median": 0.8399,
    "p10": 0.4648,
    "low_confidence_pixel_share": 0.1376,
    "flip_view_agreement": 0.824
  },
  "warnings": []
}
```

| Field | Description |
|---|---|
| `success` | Whether the prediction succeeded |
| `dominant_class` | The class ID (0–4) with the most pixels |
| `classes` | Per-class pixel counts and percentages |
| `damage_mask` | Base64-encoded RGBA PNG image (512×512) showing color-coded damage |
| `prediction_quality` | Mean maximum softmax confidence (0–100%). This is a **confidence indicator, not an accuracy measure** — softmax confidence is known to be overconfident and no ground truth exists for uploaded images |
| `confidence_details` | Additive diagnostics (ignored by the frontend): confidence mean/median/p10, share of low-confidence pixels (<0.5), and the agreement between the original and flipped inference views |
| `warnings` | Non-fatal input diagnostics — e.g. identical before/after files, or strongly mismatched aspect ratios (contents may not be spatially aligned) |

---

## Limitations

- **Unverified class labels** — The mapping of class IDs to human-readable names (No Damage, Minor, Moderate, Severe, Complete Destruction) is defined only in the frontend. Training metadata is not included in this repository, so these labels cannot be independently verified.
- **CPU inference** — The model runs on CPU and takes roughly 2–4 seconds per prediction (two forward passes with horizontal-flip test-time augmentation). GPU support is available via PyTorch CUDA but not yet configured.
- **AI-assisted only** — Results are generated by an AI model and should **not** replace professional disaster inspection or on-the-ground assessment.
- **Pixel percentages ≠ real-world area** — The reported percentages represent image pixels, not real-world square meters or geographic area.
- **Fixed resolution** — Images are resized to 512×512 before processing, which may lose fine detail in very large images.
- **Unverified training normalization** — ImageNet normalization was adopted based on checkpoint evidence (fine-tuned ImageNet-pretrained encoder) and behavioral evaluation (`scripts/evaluate_inference.py`), not on training code, which is not included in this repository.
- **Appearance sensitivity on identical pairs** — If identical files are uploaded as both before and after, the model can still report damage classes based on image appearance; the API returns a `warnings` entry in that case.

---

## Research & Evaluation

This section describes the scientific evaluation that should be performed before this project is used in production or cited in academic work. **No accuracy or performance claims are made at this stage.**

### Metrics to Evaluate

| Metric | What It Measures |
|---|---|
| **IoU (Intersection over Union)** | Overlap between predicted and ground-truth segmentation masks per class |
| **Dice Coefficient (F1)** | Harmonic mean of precision and recall for each damage class |
| **Precision** | Fraction of correctly predicted damage pixels among all predicted damage pixels |
| **Recall (Sensitivity)** | Fraction of correctly predicted damage pixels among all actual damage pixels |
| **Overall Accuracy** | Fraction of all correctly classified pixels |
| **Per-class Accuracy** | Accuracy computed independently for each of the 5 classes |
| **Inference Time** | Time taken to process one image pair (CPU vs GPU) |

### What Is Needed for Evaluation

1. **Annotated dataset** — A set of before/after image pairs with pixel-level ground-truth segmentation masks, annotated by domain experts
2. **Train/validation/test split** — To measure generalization to unseen data
3. **Standardized evaluation protocol** — Consistent preprocessing, resolution, and metric computation across all classes
4. **Baseline comparison** — Comparison against simpler approaches (e.g., image differencing, threshold-based methods) to establish the value of the deep learning approach
5. **Statistical significance** — Multiple runs or cross-validation folds to report mean and standard deviation

### Dataset Considerations

- The training dataset used for `best_model.pth` is not documented in this repository
- Class distribution in the training set may be imbalanced (e.g., "No Damage" pixels are typically far more common than "Complete Destruction")
- Geographic diversity, disaster types, and image sources should be documented
- The class label mapping (0–4) has not been verified against training metadata

### Research Directions

- **Transfer learning** — Fine-tuning the model on specific disaster types (floods, earthquakes, wildfires)
- **Multi-scale analysis** — Processing at multiple resolutions to capture both fine detail and large-scale patterns
- **Temporal analysis** — Using more than two time points for change detection
- **Uncertainty estimation** — Monte Carlo dropout or ensemble methods to provide per-pixel confidence intervals
- **Domain adaptation** — Adapting the model to new geographic regions or satellite imagery sources

---

## Future Improvements

- **Cloud deployment** — Deploy the backend on Alibaba Cloud (ECS, PAI, or Function Compute) for public access
- **GPU inference** — Enable CUDA acceleration for faster predictions
- **Verified training metadata** — Include training code, dataset documentation, and validated class label definitions
- **Real-world area estimation** — Integrate geospatial data to convert pixel percentages to approximate square meters
- **Stronger confidence metrics** — Add model confidence scores or uncertainty estimates per prediction
- **More test cases** — Expand the test image library with diverse disaster types (floods, earthquakes, wildfires)
- **Multi-language support** — Localize the frontend for different regions
- **Batch processing** — Allow uploading and analyzing multiple image pairs at once

---

## Hackathon

This project is being developed for the **Alibaba Cloud AI Hackathon Pakistan 2026**.

It demonstrates the use of AI and cloud-ready architecture for disaster response and damage assessment — a critical real-world problem where speed and accuracy can save lives and resources.

---

*Built with PyTorch, FastAPI, and segmentation_models_pytorch.*
