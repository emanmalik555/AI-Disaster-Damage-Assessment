from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import numpy as np
import base64
import io
import sys

from backend.model import model, DEVICE


app = FastAPI(
    title="Disaster Damage AI",
    description="AI-powered disaster damage assessment",
    version="1.0"
)

# Allow the frontend (opened as a local file or from any origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum upload size per file: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Inference configuration
# --------------------------------------------------------------------------- #

# Input resolution the model was trained with.
INPUT_SIZE = 512

# Resampling filter for the resize step. BICUBIC is what Pillow uses by
# default for RGB images (previous versions of this file relied on that
# default implicitly); it is spelled out here to make the behavior explicit.
RESAMPLE = Image.Resampling.BICUBIC

# ImageNet channel statistics used for input normalization.
#
# Why ImageNet statistics: the checkpoint in backend/best_model.pth was
# fine-tuned from an ImageNet-pretrained mit_b3 encoder -- backend/model.py
# builds the 6-channel first convolution by duplicating the pretrained RGB
# weights into both halves, and the first-convolution weights stored in the
# checkpoint keep the magnitude typical of ImageNet-pretrained encoders
# (std ~= 0.055; a [0,1]-trained first conv would be roughly 4x larger).
# ImageNet-pretrained encoders are trained with the statistics below.
#
# This was verified behaviorally with the offline evaluation script
# (scripts/evaluate_inference.py): with the previous plain [0,1] scaling the
# model degenerated to a near-constant prediction (~97-100% class_0 with a
# 0.9998 median softmax confidence) no matter what was uploaded -- heavily
# damaged pairs, pristine pairs, even a pair of identical files -- while
# ImageNet statistics produce spatially structured predictions that follow
# the visible damage in the test images.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Test-time augmentation: run the model on the input and on a horizontally
# flipped copy, then average the two probability maps (the flipped output is
# flipped back first). A horizontal mirror is a valid view of both nadir
# (satellite/aerial) imagery and ordinary photographs. On the bundled test
# pairs the two views agree on only 82-91% of pixels, and averaging measurably
# reduces isolated speckle. Cost: one extra forward pass (~1.3 s on CPU).
TTA_HORIZONTAL_FLIP = True

# Conservative speck removal (see remove_speck_noise). Measured on the test
# pairs: rewrites 0.02-0.03% of pixels and shifts class percentages by less
# than 0.1 percentage points, so the class distribution is preserved.
SPECK_FILTER = True
SPECK_MAX_OWN_COUNT = 3

# Relative aspect-ratio difference between the before and after image that
# triggers an alignment warning. Both images are resized to the same square
# model input independently, so very different aspect ratios mean the two
# images are stretched differently and their contents may no longer line up
# pixel-to-pixel.
ASPECT_RATIO_TOLERANCE = 0.10


@app.get("/")
def home():
    return {
        "message": "Disaster Damage AI API is running",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "num_classes": 5,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }


def preprocess_image(image: Image.Image):
    """
    Convert image to RGB and prepare it for the model.
    Corrects EXIF orientation so rotated photos (e.g. from phones) are handled properly.

    Steps:
      1. EXIF orientation correction.
      2. Conversion to RGB (handles grayscale, palette and CMYK uploads).
      3. Resize to the 512x512 model input resolution (BICUBIC).
      4. Scale to [0,1] and normalize with ImageNet channel statistics
         (see IMAGENET_MEAN / IMAGENET_STD above for the justification).
    """

    # Fix EXIF orientation before converting to RGB
    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    # Resize both images to the same size.
    image = image.resize((INPUT_SIZE, INPUT_SIZE), RESAMPLE)

    image_array = np.array(image).astype(np.float32) / 255.0

    # Normalize with ImageNet channel statistics.
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    image_array = np.transpose(image_array, (2, 0, 1))

    return torch.tensor(
        np.ascontiguousarray(image_array),
        dtype=torch.float32
    )


def run_inference(combined: torch.Tensor):
    """
    Run the segmentation model on a (1, 6, H, W) input tensor.

    Returns:
      probabilities  -- (1, 5, H, W) softmax probabilities over the 5 classes.
                       With TTA_HORIZONTAL_FLIP enabled this is the average of
                       the original view and the horizontally flipped view.
      view_agreement -- fraction of pixels where the original-view and
                       flipped-view argmax predictions agree (1.0 when TTA is
                       disabled). A low agreement means the prediction is
                       sensitive to viewpoint and deserves extra caution; it is
                       NOT an accuracy measure.
    """
    with torch.inference_mode():
        probs = F.softmax(model(combined), dim=1)

        if not TTA_HORIZONTAL_FLIP:
            return probs, 1.0

        probs_flipped = torch.flip(
            F.softmax(model(torch.flip(combined, dims=[3])), dim=1),
            dims=[3]
        )
        agreement = float(
            (probs.argmax(dim=1) == probs_flipped.argmax(dim=1))
            .float()
            .mean()
            .item()
        )
        return 0.5 * (probs + probs_flipped), agreement


def remove_speck_noise(prediction: np.ndarray) -> np.ndarray:
    """
    Single conservative 3x3 majority pass over the predicted classes.

    A pixel is rewritten only when its own class appears at most
    SPECK_MAX_OWN_COUNT times in its 3x3 neighborhood (including itself)
    while another class is strictly more frequent there. With the default
    threshold of 3 this removes isolated 1-3 pixel specks; every 2x2 block
    survives because each of its pixels counts 4 same-class pixels in its
    neighborhood. Exact ties between other classes resolve to the lower class
    id, which only affects rare boundary cases.
    """
    h, w = prediction.shape
    flat = prediction.ravel()

    counts = np.zeros((h * w, 5), dtype=np.int16)
    padded = np.pad(prediction, 1, mode="edge")
    rows = np.arange(h * w)

    for dy in range(3):
        for dx in range(3):
            window = padded[dy:dy + h, dx:dx + w].ravel()
            counts[rows, window] += 1

    own_count = counts[rows, flat]
    majority = counts.argmax(axis=1)
    majority_count = counts.max(axis=1)

    replace = (own_count <= SPECK_MAX_OWN_COUNT) & (majority_count > own_count)
    flat = np.where(replace, majority, flat)

    return flat.reshape(h, w)


DAMAGE_COLORS = {
    0: (46, 204, 113, 0),
    1: (241, 196, 15, 180),
    2: (230, 126, 34, 180),
    3: (231, 76, 60, 180),
    4: (142, 68, 173, 180),
}


def create_damage_mask(prediction: np.ndarray) -> str:
    h, w = prediction.shape
    mask = np.zeros((h, w, 4), dtype=np.uint8)

    for class_id, color in DAMAGE_COLORS.items():
        mask[prediction == class_id] = color

    mask_image = Image.fromarray(mask, mode="RGBA")

    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/predict")
async def predict(
    before: UploadFile = File(...),
    after: UploadFile = File(...)
):

    try:
        # Read uploaded images
        before_bytes = await before.read()
        after_bytes = await after.read()

        if not before_bytes or not after_bytes:
            raise HTTPException(
                status_code=400,
                detail="One or both uploaded files are empty."
            )

        # Enforce file size limit (10 MB per file)
        if len(before_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"The 'before' image is too large ({len(before_bytes) / (1024*1024):.1f} MB). Maximum allowed size is 10 MB."
            )
        if len(after_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"The 'after' image is too large ({len(after_bytes) / (1024*1024):.1f} MB). Maximum allowed size is 10 MB."
            )

        try:
            before_image = Image.open(io.BytesIO(before_bytes))
            after_image = Image.open(io.BytesIO(after_bytes))
            before_image.verify()
            after_image.verify()
            before_image = Image.open(io.BytesIO(before_bytes))
            after_image = Image.open(io.BytesIO(after_bytes))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="One or both files are not valid images. Please upload JPG, PNG, or WEBP."
            )

        # Non-fatal input diagnostics (returned in the response as warnings;
        # they do not change the prediction itself).
        warnings = []

        if before_bytes == after_bytes:
            warnings.append(
                "The 'before' and 'after' uploads are identical files, so no "
                "change between the two images exists. Damage predictions made "
                "on an identical pair reflect image appearance only and are not "
                "reliable."
            )

        before_prepared = ImageOps.exif_transpose(before_image).convert("RGB")
        after_prepared = ImageOps.exif_transpose(after_image).convert("RGB")

        ratio_before = before_prepared.width / before_prepared.height
        ratio_after = after_prepared.width / after_prepared.height
        ratio_diff = abs(ratio_before - ratio_after) / max(ratio_before, ratio_after)
        if ratio_diff > ASPECT_RATIO_TOLERANCE:
            warnings.append(
                f"The before and after images have noticeably different aspect "
                f"ratios ({ratio_before:.2f} vs {ratio_after:.2f}). Both are "
                f"resized to the same square model input, so their contents may "
                f"not be spatially aligned."
            )

        # Convert images to tensors
        before_tensor = preprocess_image(before_prepared)
        after_tensor = preprocess_image(after_prepared)

        # Combine:
        # before = 3 channels
        # after  = 3 channels
        # total  = 6 channels
        combined = torch.cat(
            [before_tensor, after_tensor],
            dim=0
        )

        # Add batch dimension
        combined = combined.unsqueeze(0).to(DEVICE)

        # Run model (optionally with horizontal-flip TTA)
        probabilities, view_agreement = run_inference(combined)

        # Convert probabilities to the per-pixel class map (argmax over the
        # class dimension), then remove isolated 1-3 pixel specks.
        prediction = probabilities.argmax(dim=1).squeeze(0).cpu().numpy()

        if SPECK_FILTER:
            prediction = remove_speck_noise(prediction)

        # Count pixels belonging to each class
        total_pixels = prediction.size

        class_counts = {}

        for class_id in range(5):
            count = int(np.sum(prediction == class_id))
            percentage = (count / total_pixels) * 100

            class_counts[f"class_{class_id}"] = {
                "pixels": count,
                "percentage": round(percentage, 2)
            }

        # Overall dominant class
        unique, counts = np.unique(
            prediction,
            return_counts=True
        )

        dominant_class = int(
            unique[np.argmax(counts)]
        )

        # Prediction quality: mean maximum softmax probability over all pixels,
        # expressed in percent. This is a CONFIDENCE indicator ("how sure is
        # the model of its own per-pixel choice"), NOT an accuracy measure --
        # softmax confidence is known to be overconfident, and no ground truth
        # exists for uploaded images. With TTA enabled the averaged
        # probabilities are used, so disagreement between the original and the
        # flipped view lowers the reported confidence honestly.
        max_probs = probabilities.max(dim=1)[0].squeeze(0).cpu().numpy()
        mean_confidence = float(max_probs.mean())
        prediction_quality = round(mean_confidence * 100, 1)

        confidence_details = {
            "metric": "mean_max_softmax_confidence",
            "mean": round(mean_confidence, 4),
            "median": round(float(np.median(max_probs)), 4),
            "p10": round(float(np.percentile(max_probs, 10)), 4),
            "low_confidence_pixel_share": round(
                float((max_probs < 0.5).mean()), 4
            ),
            "flip_view_agreement": (
                round(view_agreement, 4) if TTA_HORIZONTAL_FLIP else None
            ),
        }

        damage_mask = create_damage_mask(prediction)

        return {
            "success": True,
            "dominant_class": dominant_class,
            "classes": class_counts,
            "damage_mask": damage_mask,
            "prediction_quality": prediction_quality,
            # Additive diagnostic fields (the frontend does not read these):
            "confidence_details": confidence_details,
            "warnings": warnings
        }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
