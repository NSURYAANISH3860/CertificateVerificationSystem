from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from main.core.baseline import TemplateCluster, classify_regions, extract_fields, build_template_profile
from main.core.io import load_document_images
from main.core.ocr import get_ocr_engine
from main.core.preprocessing import preprocess_page
from main.core.schemas import DocumentType, ExtractionOutput, OcrBox, QualityReport
from main.core.settings import DEFAULT_INSTITUTION, DEFAULT_OCR_ENGINE, OUTPUT_DIR, ensure_runtime_dirs
from main.core.tables import extract_marksheet_tables, extract_marksheet_tables_with_ppstructure
from main.core.vocabulary import ControlledVocabulary


def process_document(
    file_path: str | Path,
    *,
    document_type: DocumentType | str = DocumentType.UNKNOWN,
    institution: str = DEFAULT_INSTITUTION,
    template_id: str | None = None,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    ocr_lang: str = "eng",
    save: bool = True,
    claimed_year: int | None = None,
) -> ExtractionOutput:
    ensure_runtime_dirs()
    path = Path(file_path).expanduser().resolve()
    if not template_id and claimed_year:
        clean_inst = "".join(c if c.isalnum() else "_" for c in institution)
        template_id = f"{clean_inst}_{claimed_year}".upper()
    doc_type = document_type if isinstance(document_type, DocumentType) else DocumentType(document_type)
    document_id = make_document_id(path)
    engine = get_ocr_engine(ocr_engine, lang=ocr_lang)
    vocabulary = ControlledVocabulary()

    pages = load_document_images(path)
    all_boxes: list[OcrBox] = []
    page_qualities: list[QualityReport] = []
    preprocessed_pages = []
    warnings: list[str] = []

    for page in pages:
        processed = preprocess_page(page.image)
        preprocessed_pages.append((page.page_number, processed.image))
        page_qualities.append(processed.quality)
        try:
            boxes = engine.run(processed.image, document_id=document_id, page_number=page.page_number)
        except Exception as exc:
            if engine.name != "tesseract":
                warnings.append(f"{engine.name} failed on page {page.page_number}: {exc}. Retrying with Tesseract.")
                fallback = get_ocr_engine("tesseract", lang=ocr_lang)
                boxes = fallback.run(processed.image, document_id=document_id, page_number=page.page_number)
                engine = fallback
            else:
                raise
        all_boxes.extend(boxes)

    quality = aggregate_quality(page_qualities)
    
    if template_id and all_boxes:
        save_document_boxes(template_id, document_id, all_boxes)
        rebuild_template_profile(template_id)
        
    template_profile = load_template_profile(template_id)
    regions = classify_regions(all_boxes, vocabulary=vocabulary, quality=quality, template_profile=template_profile)
    fields = extract_fields(all_boxes, regions, vocabulary=vocabulary, quality=quality)
    tables = []
    if doc_type == DocumentType.MARKSHEET:
        try:
            tables = extract_marksheet_tables_with_ppstructure(preprocessed_pages)
        except Exception as exc:
            warnings.append(f"PP-Structure table extraction unavailable: {exc}. Falling back to OCR line grouping.")
        if not tables:
            tables = extract_marksheet_tables(all_boxes)

    verification_report = None
    if claimed_year is not None and preprocessed_pages:
        from main.core.fraud import verify_certificate
        verification_report = verify_certificate(
            image=preprocessed_pages[0][1],
            boxes=all_boxes,
            fields=fields,
            claimed_year=claimed_year,
            institution=institution,
            doc_path=path,
        )

    output = ExtractionOutput(
        document_id=document_id,
        institution=institution,
        document_type=doc_type,
        template_id=template_id,
        source_file=str(path),
        fields=fields,
        regions=regions,
        tables=tables,
        quality=quality,
        ocr_engine=engine.name,
        warnings=warnings,
        audit={
            "page_count": len(pages),
            "ocr_box_count": len(all_boxes),
            "field_count": len(fields),
            "table_count": len(tables),
            "template_profile_used": bool(template_profile),
        },
        verification=verification_report,
    )
    if save:
        save_extraction(output)
    return output


def save_extraction(output: ExtractionOutput, output_dir: str | Path = OUTPUT_DIR) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{output.document_id}.json"
    target.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    return target


def make_document_id(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.name).encode("utf-8"))
    try:
        with path.open("rb") as fh:
            digest.update(fh.read(1024 * 1024))
    except OSError:
        pass
    return f"DOC_{digest.hexdigest()[:12].upper()}"


def aggregate_quality(reports: list[QualityReport]) -> QualityReport:
    if not reports:
        return QualityReport(
            blur_score=1.0,
            skew_deg=0.0,
            low_contrast=True,
            contrast_score=0.0,
            resolution_score=0.0,
            overall_quality=0.0,
            notes=["no_pages"],
        )

    def avg(attr: str) -> float:
        return sum(float(getattr(report, attr)) for report in reports) / len(reports)

    notes = sorted({note for report in reports for note in report.notes})
    return QualityReport(
        blur_score=avg("blur_score"),
        skew_deg=avg("skew_deg"),
        low_contrast=any(report.low_contrast for report in reports),
        contrast_score=avg("contrast_score"),
        resolution_score=avg("resolution_score"),
        compression_score=avg("compression_score"),
        shadow_score=avg("shadow_score"),
        glare_score=avg("glare_score"),
        crop_risk_score=avg("crop_risk_score"),
        overall_quality=avg("overall_quality"),
        notes=notes,
    )


def load_template_profile(template_id: str | None) -> list[Any] | None:
    if not template_id:
        return None
    profile_path = Path("models") / "templates" / f"{template_id}.json"
    if not profile_path.exists():
        return None
    with profile_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    clusters: list[TemplateCluster] = []
    for item in raw:
        payload = dict(item)
        payload["bucket"] = tuple(payload["bucket"])
        clusters.append(TemplateCluster(**payload))
    return clusters


def save_document_boxes(template_id: str, document_id: str, boxes: list[OcrBox]) -> None:
    from main.core.settings import DATA_DIR
    template_dir = DATA_DIR / "templates" / template_id
    template_dir.mkdir(parents=True, exist_ok=True)
    target = template_dir / f"{document_id}.json"
    
    serialized = [box.model_dump(mode="json") for box in boxes]
    target.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def rebuild_template_profile(template_id: str) -> None:
    from main.core.settings import DATA_DIR, TEMPLATE_DIR
    from main.core.baseline import build_template_profile
    from dataclasses import asdict
    
    template_dir = DATA_DIR / "templates" / template_id
    if not template_dir.exists():
        return
        
    files = list(template_dir.glob("*.json"))
    if len(files) < 2:
        return
        
    documents = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            docs_boxes = [OcrBox(**item) for item in raw]
            documents.append(docs_boxes)
        except Exception as exc:
            pass
            
    if not documents:
        return
        
    profile = build_template_profile(documents, repetition_threshold=0.85)
    target = TEMPLATE_DIR / f"{template_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(cluster) for cluster in profile], indent=2), encoding="utf-8")
