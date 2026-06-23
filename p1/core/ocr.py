from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from p1.core.io import normalized_bbox
from p1.core.schemas import OcrBox

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
    if selected != "auto":
        raise ValueError(f"Unknown OCR engine: {engine}")
    try:
        return PaddleOcrEngine(lang="en" if lang == "eng" else lang)
    except Exception as exc:
        logger.warning("PaddleOCR unavailable, falling back to Tesseract: %s", exc)
        return TesseractOcrEngine(lang=lang)


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
