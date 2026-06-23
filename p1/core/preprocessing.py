from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from p1.core.schemas import QualityReport


@dataclass(slots=True)
class PreprocessedPage:
    image: Image.Image
    quality: QualityReport
    width: int
    height: int


def preprocess_page(image: Image.Image, *, deskew: bool = True) -> PreprocessedPage:
    rgb = image.convert("RGB")
    original = np.array(rgb)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    skew_deg = estimate_skew_deg(gray)
    if deskew and abs(skew_deg) >= 0.4 and abs(skew_deg) <= 15:
        original = rotate_image(original, skew_deg)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    contrast_score = compute_contrast_score(gray)
    low_contrast = contrast_score < 0.22
    if low_contrast:
        gray = normalize_contrast(gray)

    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

    blur_score = compute_blur_score(gray)
    resolution_score = compute_resolution_score(rgb.width, rgb.height)
    shadow_score = estimate_shadow_score(gray)
    glare_score = estimate_glare_score(gray)
    crop_risk_score = estimate_crop_risk(gray)
    compression_score = estimate_compression_score(gray)

    good_quality = np.mean(
        [
            1.0 - blur_score,
            contrast_score,
            resolution_score,
            1.0 - shadow_score,
            1.0 - glare_score,
            1.0 - crop_risk_score,
            1.0 - compression_score,
        ]
    )
    notes: list[str] = []
    if blur_score > 0.65:
        notes.append("high_blur_risk")
    if low_contrast:
        notes.append("low_contrast")
    if resolution_score < 0.45:
        notes.append("low_resolution")
    if abs(skew_deg) > 3:
        notes.append("skew_detected")

    quality = QualityReport(
        blur_score=float(blur_score),
        skew_deg=float(skew_deg),
        low_contrast=bool(low_contrast),
        contrast_score=float(contrast_score),
        resolution_score=float(resolution_score),
        compression_score=float(compression_score),
        shadow_score=float(shadow_score),
        glare_score=float(glare_score),
        crop_risk_score=float(crop_risk_score),
        overall_quality=float(max(0.0, min(1.0, good_quality))),
        notes=notes,
    )
    out = Image.fromarray(processed)
    return PreprocessedPage(image=out, quality=quality, width=out.width, height=out.height)


def compute_blur_score(gray: np.ndarray) -> float:
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return max(0.0, min(1.0, 1.0 - (variance / 900.0)))


def compute_contrast_score(gray: np.ndarray) -> float:
    p5, p95 = np.percentile(gray, [5, 95])
    return float(max(0.0, min(1.0, (p95 - p5) / 255.0)))


def compute_resolution_score(width: int, height: int) -> float:
    megapixels = (width * height) / 1_000_000.0
    return float(max(0.0, min(1.0, megapixels / 2.0)))


def estimate_shadow_score(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return float(max(0.0, min(1.0, np.std(small) / 90.0)))


def estimate_glare_score(gray: np.ndarray) -> float:
    bright_ratio = float(np.mean(gray > 245))
    return max(0.0, min(1.0, bright_ratio * 4.0))


def estimate_crop_risk(gray: np.ndarray) -> float:
    border = max(5, min(gray.shape[:2]) // 50)
    edges = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )
    dark_border_ratio = float(np.mean(edges < 35))
    return max(0.0, min(1.0, dark_border_ratio * 2.0))


def estimate_compression_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    return max(0.0, min(1.0, max(0.0, edge_density - 0.18) * 3.0))


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_skew_deg(gray: np.ndarray) -> float:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
