from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CVS_DATA_DIR", PROJECT_ROOT / "data"))
LOOKUP_DIR = Path(os.getenv("CVS_LOOKUP_DIR", DATA_DIR / "lookups"))
OUTPUT_DIR = Path(os.getenv("CVS_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
MODEL_DIR = Path(os.getenv("CVS_MODEL_DIR", PROJECT_ROOT / "models"))
TEMPLATE_DIR = Path(os.getenv("CVS_TEMPLATE_DIR", MODEL_DIR / "templates"))

DEFAULT_INSTITUTION = os.getenv("CVS_DEFAULT_INSTITUTION", "JNTUH")
DEFAULT_REVIEW_THRESHOLD = float(os.getenv("CVS_REVIEW_THRESHOLD", "0.75"))
DEFAULT_OCR_ENGINE = os.getenv("CVS_OCR_ENGINE", "auto")

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_DOCUMENT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, LOOKUP_DIR, OUTPUT_DIR, MODEL_DIR, TEMPLATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
