# 60–90 Second Hackathon Demo Guide — Disaster Damage AI

> Presenter cheat sheet for the Alibaba Cloud AI Hackathon Pakistan 2026 live demo.
> Test pair: **Test Case 2** (`before2.jpg` → `after2.jpg`) — chosen because it gives the most
> visually meaningful result (MODERATE DAMAGE, 15% affected, clear overlay contrast).
> Verified working: backend API + full browser flow re-tested on September 2026 (all 12 checkpoints passed).

---

## Pre-Demo Checklist (do 5 minutes before going on stage)

1. Backend running: open `http://127.0.0.1:8000/health` — must show `"status": "healthy"`, `"model_loaded": true`.
   Or run: `.venv\Scripts\python.exe scripts\demo_check.py` (posts Test Case 2 and prints the expected numbers).
2. Browser open at `frontend/index.html`, scrolled to the top, at ~110–125% zoom (judges must read the numbers).
3. File Explorer already open in the `test_images` folder (so the upload dialogs open instantly).
4. Notifications silenced, screen sleep disabled, other tabs closed.
5. Fallback screenshots ready in `test_images\demo2_*.png` (worst-case backup).

---

## The 60–90 Second Demo Flow

| # | Time | Action (exact) | Screen / focus | What to say (short version) |
|---|------|----------------|---------------|------------------------------|
| 1 | 0:00–0:10 | Open the app (already loaded, top of page). No clicks — gesture at the hero + 3 feature cards. | Hero: "Disaster Damage AI" + feature cards | Problem/solution pitch (see script below) |
| 2 | 0:10–0:17 | Click the **Test Case 2** card → alert shows the built-in demo steps → click OK. | Quick Demo Guide section | "Ships with a built-in demo guide — I'll use Test Case 2." |
| 3 | 0:17–0:25 | Click left upload box (**Before Disaster**) → select `before2.jpg`. | Upload box + live preview | "The area before the disaster." |
| 4 | 0:25–0:33 | Click right upload box (**After Disaster**) → select `after2.jpg`. | Upload box + live preview | "Same location, after the disaster." |
| 5 | 0:33–0:36 | Click **🔍 Analyze Damage**. Keep the button in view. | Button + spinner "Analyzing Damage..." | "Now I analyze." |
| 6 | 0:36–0:40 | Point at the spinner/loading state (lasts ~2.5 s on CPU). | Analyze button | "Running live on CPU in just a couple of seconds." |
| 7 | 0:40–0:50 | Scroll to results. Point at badge, then the two big stats. | Overall Assessment: MODERATE DAMAGE badge, 15.0% vs 85.0% | "Moderate damage — 15% affected, 85% intact." |
| 8 | 0:50–0:55 | Point at the quality bar. | Prediction Quality (softmax confidence) — 88% | "88% softmax confidence — the model is sure." |
| 9 | 0:55–1:03 | Point at the 5 class bars top-to-bottom. | Class Distribution (5 classes) | "Every pixel classified into 5 levels." |
| 10 | 1:03–1:10 | Point at the colored regions on the canvas. | Damage Overlay canvas | "The damage map shows exactly where." |
| 11 | 1:10–1:17 | Drag the opacity slider 60 → 80 (a single decisive drag). | Overlay slider + canvas change | "Adjustable overlay to verify against real imagery." |
| 12 | 1:17–1:21 | Scroll one screen down. | Before vs After side-by-side | "The before/after view confirms the change." |
| 13 | 1:21–1:26 | Point at the download button (clicking is safe — it saves an HTML report instantly). | 📥 Download Analysis Report | "One click — a shareable HTML report for response teams." |
| 14 | 1:26–1:30 | Scroll to the bottom. | AI-Assisted Assessment disclaimer + footer | "AI-assisted decision support — not a replacement for experts." |

Total: ~90 seconds at a calm pace. To land at ~70 seconds, drop the "running live on CPU" line and the Before-vs-After beat — the pitch, upload, analysis, assessment, overlay and report are the must-haves.

---

## Exact Speaking Script (~90 seconds, with actions)

**[0:00 — hero + feature cards, no clicks]**
"After a disaster, responders need one answer fast: how bad is it, and where? Disaster Damage AI answers that in seconds. We take before-and-after imagery, and a deep-learning segmentation model classifies every single pixel by damage level."

**[0:10 — click Test Case 2, OK the alert]**
"The app even ships with a built-in demo guide. I'll use Test Case 2 — a real disaster scene."

**[0:17 — upload before2.jpg]**
"First, the before image — this area before the event."

**[0:25 — upload after2.jpg]**
"And the after image — the same location, after the disaster."

**[0:33 — click Analyze Damage]**
"Now I simply analyze."

**[0:36 — loading spinner]**
"This is running live on CPU — it takes just a couple of seconds."

**[0:40 — Overall Assessment]**
"Here's the overall assessment: MODERATE damage. Fifteen percent of the area is affected, eighty-five percent is intact — and the model reports 88 percent softmax confidence, so it's sure about what it sees."

**[0:55 — Class Distribution]**
"Every pixel falls into one of five damage classes — most of the damage here is minor, about six percent, with nearly nine percent complete destruction."

**[1:03 — Damage Overlay]**
"The damage map pinpoints exactly where — and I can adjust the overlay opacity to check it against the underlying imagery."

**[1:17 — Before vs After]**
"The before-and-after comparison confirms the change —"

**[1:21 — Download button]**
"— and one click exports this whole analysis as a shareable HTML report for response teams."

**[1:26 — disclaimer/footer]**
"And importantly: this is AI-assisted decision support — built to speed up, not replace, professional inspection. Disaster Damage AI — from images to insight in seconds."

*(If running over time: drop the "running live on CPU" line and the Before-vs-After line to land at ~70 seconds.)*

---

## Expected Numbers (Test Case 2) — know these cold

| Metric | Value |
|---|---|
| Assessment badge | ⚠️ MODERATE DAMAGE |
| Total Affected Area | 15.0% |
| Unaffected Area | 85.0% |
| Prediction Quality (softmax confidence) | 88.0% |
| No Damage | 85.0% |
| Minor Damage | 6.2% (dominant damage class) |
| Moderate Damage | 0.0% |
| Severe Damage | 0.0% |
| Complete Destruction | 8.8% |
| Damage Types Detected | 2 of 4 |
| Inference time | ~2.4 s (CPU) |

---

## What NOT to Click / Do During the Demo

- **Do NOT click Analyze Damage before both images are uploaded** — it shows the validation error (fine in a longer talk; wastes time here).
- **Do NOT click the Test Case 1 card** — its alert and instructions point to the wrong pair.
- **Do NOT refresh or navigate away mid-demo** — results are in-memory only and will be lost.
- **Do NOT click Analyze Damage twice / while loading** — the button disables itself, but a second click after results re-runs inference and wastes ~3 s.
- **Do NOT upload arbitrary or identical images** — unverified inputs can trigger warnings; Test Case 2 is the rehearsed, predictable result.
- **Do NOT open DevTools, other tabs, or resize to mobile** — nothing is wrong with them; they just steal seconds.
- **Do NOT start the demo before `/health` shows `model_loaded: true`** — otherwise the first prediction includes ~5 s of model loading.

---

## Backup Plans

### A. Backend is already running (the normal case)
That is the desired state — a warm backend means the model is already loaded and the first prediction takes ~2.5 s instead of ~7 s.
1. Verify `http://127.0.0.1:8000/health` → `"status": "healthy"`.
2. If it responds but behaves oddly (stale process from an older code version): stop it, then restart:
   `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` and wait ~5 s before demoing.

### B. Analysis takes longer than expected
Normal CPU round-trip is ~2.4–2.6 s. If the spinner is still running:
1. **Keep talking** — the loading state is on-screen proof the model is live; narrate the pipeline (upload → 6-channel tensor → U-Net → pixel classes → overlay) while it finishes.
2. **Never refresh and never re-click the button** — the request is still in flight; a refresh loses everything and re-pays the model load.
3. If it exceeds ~20 s: apologize once ("model loading on first call — one moment"), let it finish. The UI auto-recovers when the response arrives.
4. If it actually errors: restart the backend (command above), re-upload the same two images, re-analyze — rehearsed recovery, ~15 s.

### C. Backend will not start at all (worst case)
Present the fallback screenshots in order — `test_images\demo2_01_initial_page.png` → `demo2_05_loading_state.png` → `demo2_06_overall_assessment.png` → `demo2_08_overlay_80.png` → `demo2_09_before_after.png` → `demo2_11_full_results.png` — while delivering the same script. The story and the numbers are identical.

---

## Quick Reference — Start Commands

```powershell
# Backend (must be running before the demo)
cd "c:\Users\Eman Malik\Disaster-Damage-AI"
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 30-second pre-flight check (posts Test Case 2 to the live API)
python scripts\demo_check.py

# Frontend — open in browser
frontend\index.html
```
