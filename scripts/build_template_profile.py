from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.baseline import build_template_profile
from main.core.io import load_document_images
from main.core.ocr import get_ocr_engine
from main.core.pipeline import make_document_id
from main.core.preprocessing import preprocess_page
from main.core.settings import TEMPLATE_DIR, ensure_runtime_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a P1 template repetition profile from reference documents.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--ocr-engine", choices=["auto", "paddleocr", "tesseract"], default="auto")
    parser.add_argument("--repetition-threshold", type=float, default=0.85)
    parser.add_argument("--output-dir", type=Path, default=TEMPLATE_DIR)
    args = parser.parse_args()

    ensure_runtime_dirs()
    engine = get_ocr_engine(args.ocr_engine)
    documents = []
    for file_path in args.files:
        document_id = make_document_id(file_path.resolve())
        boxes = []
        for page in load_document_images(file_path):
            processed = preprocess_page(page.image)
            boxes.extend(engine.run(processed.image, document_id=document_id, page_number=page.page_number))
        documents.append(boxes)
        print(f"{file_path}: {len(boxes)} OCR boxes")

    profile = build_template_profile(documents, repetition_threshold=args.repetition_threshold)
    target = args.output_dir / f"{args.template_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(cluster) for cluster in profile], indent=2), encoding="utf-8")
    print(f"Wrote {len(profile)} template clusters to {target}")


if __name__ == "__main__":
    main()
