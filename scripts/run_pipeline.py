from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.pipeline import process_document, save_extraction
from main.core.schemas import DocumentType


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 extraction on one document.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--document-type", choices=[item.value for item in DocumentType], default=DocumentType.UNKNOWN.value)
    parser.add_argument("--institution", default="JNTUH")
    parser.add_argument("--template-id", default=None)
    parser.add_argument("--ocr-engine", default="auto", choices=["auto", "paddleocr", "tesseract"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    output = process_document(
        args.file,
        document_type=args.document_type,
        institution=args.institution,
        template_id=args.template_id,
        ocr_engine=args.ocr_engine,
        save=False,
    )
    target = save_extraction(output, args.output_dir)
    print(f"Wrote {target}")
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
