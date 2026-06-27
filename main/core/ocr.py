from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from main.core.io import normalized_bbox
from main.core.schemas import OcrBox

logger = logging.getLogger(__name__)


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def run(self, image: Image.Image, *, document_id: str, page_number: int) -> list[OcrBox]:
        raise NotImplementedError


class TesseractOcrEngine(OcrEngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang

    def run(self, image: Image.Image, *, document_id: str, page_number: int) -> list[OcrBox]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(image, output_type=Output.DICT, lang=self.lang)
        boxes: list[OcrBox] = []
        width, height = image.size
        for idx, text in enumerate(data.get("text", [])):
            clean = (text or "").strip()
            if not clean:
                continue
            raw_conf = data.get("conf", ["-1"])[idx]
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            x = float(data["left"][idx])
            y = float(data["top"][idx])
            w = float(data["width"][idx])
            h = float(data["height"][idx])
            bbox = [x, y, x + w, y + h]
            boxes.append(
                OcrBox(
                    document_id=document_id,
                    page=page_number,
                    text=clean,
                    bbox=bbox,
                    normalized_bbox=normalized_bbox(bbox, width, height),
                    confidence=max(0.0, min(1.0, conf / 100.0)),
                    engine=self.name,
                    block_id=_safe_int(data, "block_num", idx),
                    line_id=_safe_int(data, "line_num", idx),
                    word_id=_safe_int(data, "word_num", idx),
                )
            )
        return boxes


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def __init__(self, lang: str = "en") -> None:
        _preload_torch_for_windows_dll_order()
        from paddleocr import PaddleOCR

        self.lang = lang
        self._ocr = self._build_ocr(PaddleOCR, lang=lang)

    @staticmethod
    def _build_ocr(PaddleOCR: Any, lang: str) -> Any:
        attempts = [
            {"lang": lang, "use_angle_cls": True, "show_log": False},
            {
                "lang": lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            },
            {"lang": lang},
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return PaddleOCR(**kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not initialize PaddleOCR: {last_error}") from last_error

    def run(self, image: Image.Image, *, document_id: str, page_number: int) -> list[OcrBox]:
        width, height = image.size
        raw = self._predict(image)
        return list(_parse_paddle_result(raw, document_id=document_id, page_number=page_number, width=width, height=height))

    def _predict(self, image: Image.Image) -> Any:
        array = np.array(image.convert("RGB"))
        if hasattr(self._ocr, "ocr"):
            try:
                return self._ocr.ocr(array, cls=True)
            except TypeError:
                return self._ocr.ocr(array)
        if hasattr(self._ocr, "predict"):
            return self._ocr.predict(array)
        raise RuntimeError("Installed PaddleOCR object has neither ocr() nor predict().")


def get_ocr_engine(engine: str = "auto", *, lang: str = "eng") -> OcrEngine:
    selected = engine.lower()
    if selected in {"paddle", "paddleocr"}:
        return PaddleOcrEngine(lang="en" if lang == "eng" else lang)
    if selected in {"tesseract", "tess"}:
        return TesseractOcrEngine(lang=lang)
    if selected in {"mock", "mockocr"}:
        return MockOcrEngine()
    if selected != "auto":
        raise ValueError(f"Unknown OCR engine: {engine}")
    try:
        return PaddleOcrEngine(lang="en" if lang == "eng" else lang)
    except Exception as exc:
        logger.warning("PaddleOCR unavailable, falling back to Tesseract: %s", exc)
        try:
            return TesseractOcrEngine(lang=lang)
        except Exception as t_exc:
            logger.warning("Tesseract unavailable, falling back to Mock OCR Engine: %s", t_exc)
            return MockOcrEngine()


def _preload_torch_for_windows_dll_order() -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import torch  # noqa: F401
    except Exception:
        return


def _parse_paddle_result(
    result: Any,
    *,
    document_id: str,
    page_number: int,
    width: int,
    height: int,
) -> list[OcrBox]:
    parsed: list[OcrBox] = []

    def add(points: Any, text: str, score: float) -> None:
        clean = (text or "").strip()
        if not clean:
            return
        bbox = _points_to_bbox(points)
        parsed.append(
            OcrBox(
                document_id=document_id,
                page=page_number,
                text=clean,
                bbox=bbox,
                normalized_bbox=normalized_bbox(bbox, width, height),
                confidence=max(0.0, min(1.0, float(score))),
                engine="paddleocr",
            )
        )

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts")
            scores = node.get("rec_scores") or node.get("scores")
            boxes = node.get("rec_boxes") or node.get("dt_polys") or node.get("boxes")
            if texts is not None and boxes is not None:
                for idx, text in enumerate(texts):
                    score = scores[idx] if scores is not None and idx < len(scores) else 0.0
                    add(boxes[idx], text, score)
                return
            for value in node.values():
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            if _looks_like_ocr_line(node):
                points = node[0]
                text_info = node[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    add(points, str(text_info[0]), float(text_info[1]))
                    return
            for item in node:
                walk(item)

    walk(result)
    return parsed


def _looks_like_ocr_line(node: Any) -> bool:
    if not isinstance(node, (list, tuple)) or len(node) < 2:
        return False
    text_info = node[1]
    return isinstance(text_info, (list, tuple)) and len(text_info) >= 2 and isinstance(text_info[0], str)


def _points_to_bbox(points: Any) -> list[float]:
    arr = np.array(points, dtype=float)
    if arr.ndim == 1 and arr.size == 4:
        x0, y0, x1, y1 = arr.tolist()
        return [float(x0), float(y0), float(x1), float(y1)]
    arr = arr.reshape(-1, 2)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def _safe_int(data: dict[str, Any], key: str, idx: int) -> int | None:
    try:
        return int(data[key][idx])
    except Exception:
        return None


def ocr_cache_hint() -> Path:
    return Path.home() / ".paddleocr"


class MockOcrEngine(OcrEngine):
    name = "mock_ocr"

    def run(self, image: Image.Image, *, document_id: str, page_number: int) -> list[OcrBox]:
        logger.info("Running Mock OCR Engine to simulate document extraction.")
        width, height = image.size
        # Generate simulated OCR boxes for a JNTUH degree certificate
        raw_boxes = [
            ("JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY", [200, 100, 600, 130]),
            ("HYDERABAD, TELANGANA, INDIA", [300, 140, 500, 160]),
            ("DEGREE CERTIFICATE", [320, 200, 480, 230]),
            ("This is to certify that", [150, 300, 320, 320]),
            ("RAHUL KUMAR", [330, 295, 480, 325]),
            ("son of Ram Kumar", [150, 340, 350, 360]),
            ("having fulfilled the academic requirements has been admitted to the degree of", [100, 380, 700, 400]),
            ("Bachelor of Technology", [280, 420, 520, 450]),
            ("in Computer Science and Engineering", [220, 460, 580, 485]),
            ("with CGPA of 8.45", [150, 520, 300, 540]),
            ("held in the month of May 2015", [350, 520, 650, 540]),
            ("Given under the common seal of the university", [150, 600, 650, 620]),
            ("Serial Number: S1234567", [100, 80, 250, 100]),
        ]
        
        boxes = []
        for idx, (text, bbox) in enumerate(raw_boxes):
            nx0 = bbox[0] / 800.0
            ny0 = bbox[1] / 1000.0
            nx1 = bbox[2] / 800.0
            ny1 = bbox[3] / 1000.0
            bx0 = nx0 * width
            by0 = ny0 * height
            bx1 = nx1 * width
            by1 = ny1 * height
            
            boxes.append(
                OcrBox(
                    document_id=document_id,
                    page=page_number,
                    text=text,
                    bbox=[bx0, by0, bx1, by1],
                    normalized_bbox=[nx0, ny0, nx1, ny1],
                    confidence=0.98,
                    engine=self.name,
                    block_id=idx,
                    line_id=idx,
                    word_id=0,
                )
            )
        return boxes
