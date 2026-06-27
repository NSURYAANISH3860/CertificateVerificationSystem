from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from main.core.settings import SUPPORTED_DOCUMENT_SUFFIXES, SUPPORTED_IMAGE_SUFFIXES


@dataclass(slots=True)
class PageImage:
    page_number: int
    image: Image.Image
    source_path: Path


def validate_input_file(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
        raise ValueError(f"Unsupported input type {file_path.suffix!r}. Supported: {suffixes}")
    return file_path


def load_document_images(path: str | Path, dpi: int = 220) -> list[PageImage]:
    file_path = validate_input_file(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(file_path, dpi=dpi)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return _load_image_pages(file_path)
    raise ValueError(f"Unsupported input type: {suffix}")


def _load_image_pages(path: Path) -> list[PageImage]:
    with Image.open(path) as img:
        pages: list[PageImage] = []
        for idx, frame in enumerate(ImageSequence.Iterator(img), start=1):
            page = ImageOps.exif_transpose(frame).convert("RGB")
            pages.append(PageImage(page_number=idx, image=page.copy(), source_path=path))
        return pages


def _load_pdf_pages(path: Path, dpi: int) -> list[PageImage]:
    try:
        import fitz
        import io
        
        doc = fitz.open(str(path))
        pages: list[PageImage] = []
        for idx in range(len(doc)):
            page = doc.load_page(idx)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            pages.append(PageImage(page_number=idx + 1, image=img.copy(), source_path=path))
        return pages
    except Exception as exc:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(str(path), dpi=dpi)
            return [
                PageImage(page_number=idx, image=image.convert("RGB"), source_path=path)
                for idx, image in enumerate(images, start=1)
            ]
        except Exception:
            return _load_pdf_pages_with_pdftoppm(path, dpi=dpi)


def _load_pdf_pages_with_pdftoppm(path: Path, dpi: int) -> list[PageImage]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError(
            "PDF rendering requires either pdf2image+Poppler or a pdftoppm executable on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="p1_pdf_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            [pdftoppm, "-png", "-r", str(dpi), str(path), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
        pngs = sorted(Path(tmp).glob("page-*.png"))
        if not pngs:
            raise RuntimeError(f"pdftoppm did not produce pages for {path}")
        pages: list[PageImage] = []
        for idx, png in enumerate(pngs, start=1):
            with Image.open(png) as img:
                pages.append(PageImage(page_number=idx, image=img.convert("RGB").copy(), source_path=path))
        return pages


def normalized_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = bbox
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


def merge_bboxes(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
