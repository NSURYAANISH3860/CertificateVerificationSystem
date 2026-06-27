from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download_paddleocr(lang: str = "en") -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install ML dependencies with: python -m pip install -r requirements-ml.txt"
        ) from exc

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
            PaddleOCR(**kwargs)
            print("PaddleOCR detection/recognition models are initialized and cached.")
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize PaddleOCR: {last_error}") from last_error


def download_layoutlmv3(cache_dir: Path | None = None) -> None:
    from transformers import AutoModel, AutoProcessor

    model_name = "microsoft/layoutlmv3-base"
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    AutoProcessor.from_pretrained(model_name, apply_ocr=False, **kwargs)
    AutoModel.from_pretrained(model_name, **kwargs)
    print(f"Downloaded {model_name}.")


def download_trocr(cache_dir: Path | None = None) -> None:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_name = "microsoft/trocr-base-printed"
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    TrOCRProcessor.from_pretrained(model_name, **kwargs)
    VisionEncoderDecoderModel.from_pretrained(model_name, **kwargs)
    print(f"Downloaded {model_name}.")


def download_ppstructure(lang: str = "en") -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import torch  # noqa: F401
        from paddleocr import PPStructure
    except Exception as exc:
        raise RuntimeError(
            "PP-Structure is unavailable. Install ML dependencies with: python -m pip install -r requirements-ml.txt"
        ) from exc
    PPStructure(show_log=False, lang=lang, table=True, ocr=True, formula=False)
    print("PaddleOCR PP-Structure table models are initialized and cached.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache OCR and layout models used by CVS.")
    parser.add_argument("--skip-paddleocr", action="store_true")
    parser.add_argument("--layoutlmv3", action="store_true", help="Download LayoutLMv3 base for later token/layout fine-tuning.")
    parser.add_argument("--ppstructure", action="store_true", help="Download PaddleOCR PP-Structure table models.")
    parser.add_argument("--trocr", action="store_true", help="Download TrOCR printed-text model for OCR benchmarking.")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    errors: list[str] = []
    if not args.skip_paddleocr:
        try:
            download_paddleocr(args.lang)
        except Exception as exc:
            errors.append(f"PaddleOCR: {exc}")
    if args.layoutlmv3:
        try:
            download_layoutlmv3(args.cache_dir)
        except Exception as exc:
            errors.append(f"LayoutLMv3: {exc}")
    if args.ppstructure:
        try:
            download_ppstructure(args.lang)
        except Exception as exc:
            errors.append(f"PP-Structure: {exc}")
    if args.trocr:
        try:
            download_trocr(args.cache_dir)
        except Exception as exc:
            errors.append(f"TrOCR: {exc}")

    if errors:
        print("\nModel download completed with issues:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Model download completed.")


if __name__ == "__main__":
    main()
