from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from rapidfuzz import fuzz

from main.core.io import merge_bboxes
from main.core.schemas import (
    ClassifiedRegion,
    ConfidenceComponents,
    ExtractedField,
    LabelClass,
    OcrBox,
    QualityReport,
)
from main.core.settings import DEFAULT_REVIEW_THRESHOLD
from main.core.vocabulary import CGPA_RE, DATE_RE, HALL_TICKET_RE, ControlledMatch, ControlledVocabulary, MARK_RE


@dataclass(slots=True)
class TemplateCluster:
    page: int
    bucket: tuple[int, int]
    repetition_rate: float
    canonical_text: str
    canonical_bbox: list[float]
    count: int
    total_documents: int


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", text).lower()).strip()


def position_bucket(normalized_bbox: list[float], *, grid_size: int = 36) -> tuple[int, int]:
    x0, y0, x1, y1 = normalized_bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return (min(grid_size - 1, max(0, int(cx * grid_size))), min(grid_size - 1, max(0, int(cy * grid_size))))


def build_template_profile(
    documents: list[list[OcrBox]],
    *,
    repetition_threshold: float = 0.85,
    grid_size: int = 36,
) -> list[TemplateCluster]:
    grouped: dict[tuple[int, tuple[int, int]], list[OcrBox]] = defaultdict(list)
    for boxes in documents:
        seen_per_doc: set[tuple[int, tuple[int, int], str]] = set()
        for box in boxes:
            key = (box.page, position_bucket(box.normalized_bbox, grid_size=grid_size), normalize_text(box.text))
            if key in seen_per_doc:
                continue
            seen_per_doc.add(key)
            grouped[(key[0], key[1])].append(box)

    clusters: list[TemplateCluster] = []
    total_documents = max(1, len(documents))
    for (page, bucket), boxes in grouped.items():
        text_counter = Counter(normalize_text(box.text) for box in boxes if normalize_text(box.text))
        if not text_counter:
            continue
        canonical_norm, count = text_counter.most_common(1)[0]
        repetition_rate = count / total_documents
        if repetition_rate < repetition_threshold:
            continue
        canonical_text = next(box.text for box in boxes if normalize_text(box.text) == canonical_norm)
        clusters.append(
            TemplateCluster(
                page=page,
                bucket=bucket,
                repetition_rate=float(repetition_rate),
                canonical_text=canonical_text,
                canonical_bbox=merge_bboxes([box.normalized_bbox for box in boxes]),
                count=count,
                total_documents=total_documents,
            )
        )
    return clusters


def classify_regions(
    boxes: list[OcrBox],
    *,
    vocabulary: ControlledVocabulary,
    quality: QualityReport,
    template_profile: list[TemplateCluster] | None = None,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> list[ClassifiedRegion]:
    profile_index = {(cluster.page, cluster.bucket): cluster for cluster in template_profile or []}
    regions: list[ClassifiedRegion] = []
    for box in boxes:
        label = LabelClass.OPEN_VARIABLE
        field_hint: str | None = None
        reason_codes: list[str] = []
        controlled = vocabulary.match(box.text)
        repetition_rate = None
        template_conf = None

        cluster = profile_index.get((box.page, position_bucket(box.normalized_bbox)))
        if cluster:
            similarity = fuzz.WRatio(normalize_text(box.text), normalize_text(cluster.canonical_text)) / 100.0
            repetition_rate = cluster.repetition_rate
            template_conf = float((similarity + cluster.repetition_rate) / 2.0)
            if cluster.repetition_rate >= 0.85 and similarity >= 0.85:
                label = LabelClass.CONSTANT
                reason_codes.append("template_repetition")

        if label != LabelClass.CONSTANT and controlled:
            label = LabelClass.CONTROLLED_VARIABLE
            field_hint = controlled.field_name
            reason_codes.append(f"controlled_match:{controlled.lookup_name}")

        if label == LabelClass.OPEN_VARIABLE:
            field_hint = infer_open_field_hint(box.text)
            if field_hint:
                reason_codes.append(f"pattern:{field_hint}")
            if is_probable_table_header(box.text):
                label = LabelClass.TABLE_HEADER
                field_hint = "table_header"
                reason_codes.append("table_header_terms")

        components = ConfidenceComponents(
            ocr_confidence=box.confidence,
            layout_template_confidence=template_conf,
            repetition_rate=repetition_rate,
            controlled_match_score=controlled.score if controlled else None,
            quality_score=quality.overall_quality,
        )
        confidence = estimate_region_confidence(label, components)
        regions.append(
            ClassifiedRegion(
                text=box.text,
                label_class=label,
                bbox=box.bbox,
                normalized_bbox=box.normalized_bbox,
                page=box.page,
                confidence=confidence,
                requires_review=confidence < review_threshold,
                components=components,
                field_hint=field_hint,
                reason_codes=reason_codes,
            )
        )
    return regions


def extract_fields(
    boxes: list[OcrBox],
    regions: list[ClassifiedRegion],
    *,
    vocabulary: ControlledVocabulary,
    quality: QualityReport,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, ExtractedField]:
    fields: dict[str, ExtractedField] = {}
    lines = group_boxes_into_lines(boxes)

    for line_boxes in lines:
        text = " ".join(box.text for box in line_boxes)
        bbox = merge_bboxes([box.bbox for box in line_boxes])
        page = line_boxes[0].page
        avg_ocr = mean(box.confidence for box in line_boxes)

        _maybe_set(
            fields,
            "hall_ticket_number",
            extract_regex_value(HALL_TICKET_RE, text),
            LabelClass.OPEN_VARIABLE,
            bbox,
            page,
            avg_ocr,
            quality,
            source_text=text,
            reason="pattern:hall_ticket_number",
            review_threshold=review_threshold,
        )
        _maybe_set(
            fields,
            "cgpa",
            extract_regex_value(CGPA_RE, text),
            LabelClass.OPEN_VARIABLE,
            bbox,
            page,
            avg_ocr,
            quality,
            source_text=text,
            reason="pattern:cgpa",
            review_threshold=review_threshold,
        )
        date_match = DATE_RE.search(text)
        if date_match and any(token in text.lower() for token in ["date", "issued", "exam", "month", "year"]):
            _maybe_set(
                fields,
                "issue_or_exam_date",
                date_match.group(0),
                LabelClass.OPEN_VARIABLE,
                bbox,
                page,
                avg_ocr,
                quality,
                source_text=text,
                reason="pattern:date",
                review_threshold=review_threshold,
            )
        student_name = extract_student_name(text)
        _maybe_set(
            fields,
            "student_name",
            student_name,
            LabelClass.OPEN_VARIABLE,
            bbox,
            page,
            avg_ocr,
            quality,
            source_text=text,
            reason="pattern:student_name",
            review_threshold=review_threshold,
        )

        controlled = vocabulary.match(text)
        if controlled:
            set_controlled_field(fields, controlled, bbox, page, avg_ocr, quality, source_text=text, review_threshold=review_threshold)

        if "university" in text.lower() and "university_name" not in fields:
            confidence = bounded(mean(avg_ocr, quality.overall_quality, 0.92))
            fields["university_name"] = ExtractedField(
                value=text.strip(),
                label_class=LabelClass.CONSTANT,
                bbox=bbox,
                page=page,
                confidence=confidence,
                requires_review=confidence < review_threshold,
                components=ConfidenceComponents(ocr_confidence=avg_ocr, quality_score=quality.overall_quality, repetition_rate=0.85),
                source_text=text,
                reason_codes=["keyword:university"],
            )

    for region in regions:
        if region.label_class == LabelClass.CONTROLLED_VARIABLE and region.field_hint:
            controlled = vocabulary.match(region.text)
            if controlled:
                set_controlled_field(
                    fields,
                    controlled,
                    region.bbox,
                    region.page,
                    region.components.ocr_confidence or region.confidence,
                    quality,
                    source_text=region.text,
                    review_threshold=review_threshold,
                )

    return fields


def group_boxes_into_lines(boxes: list[OcrBox], *, y_tolerance: float = 0.012) -> list[list[OcrBox]]:
    sorted_boxes = sorted(boxes, key=lambda b: (b.page, (b.normalized_bbox[1] + b.normalized_bbox[3]) / 2, b.normalized_bbox[0]))
    lines: list[list[OcrBox]] = []
    for box in sorted_boxes:
        cy = (box.normalized_bbox[1] + box.normalized_bbox[3]) / 2
        placed = False
        for line in reversed(lines[-8:]):
            if line[0].page != box.page:
                continue
            line_cy = mean((item.normalized_bbox[1] + item.normalized_bbox[3]) / 2 for item in line)
            if abs(cy - line_cy) <= y_tolerance:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    for line in lines:
        line.sort(key=lambda b: b.normalized_bbox[0])
    return lines


def infer_open_field_hint(text: str) -> str | None:
    if HALL_TICKET_RE.search(text):
        return "hall_ticket_number"
    if CGPA_RE.search(text):
        return "cgpa"
    if DATE_RE.search(text):
        return "date"
    if MARK_RE.fullmatch(text.strip()):
        return "marks"
    return None


def is_probable_table_header(text: str) -> bool:
    lower = text.lower()
    terms = {"subject", "code", "grade", "credits", "internal", "external", "marks", "total"}
    return sum(1 for term in terms if term in lower) >= 2


def estimate_region_confidence(label: LabelClass, components: ConfidenceComponents) -> float:
    values = [components.ocr_confidence or 0.0, components.quality_score or 0.0]
    if label == LabelClass.CONSTANT:
        values.append(components.layout_template_confidence or components.repetition_rate or 0.65)
    elif label == LabelClass.CONTROLLED_VARIABLE:
        values.append(components.controlled_match_score or 0.70)
    elif label == LabelClass.TABLE_HEADER:
        values.append(0.82)
    else:
        values.append(0.68)
    return bounded(mean(values))


def set_controlled_field(
    fields: dict[str, ExtractedField],
    controlled: ControlledMatch,
    bbox: list[float],
    page: int,
    ocr_confidence: float,
    quality: QualityReport,
    *,
    source_text: str,
    review_threshold: float,
) -> None:
    field_name = controlled.field_name
    confidence = bounded(mean(ocr_confidence, quality.overall_quality, controlled.score))
    if field_name in fields and fields[field_name].confidence >= confidence:
        return
    fields[field_name] = ExtractedField(
        value=controlled.value,
        label_class=LabelClass.CONTROLLED_VARIABLE,
        bbox=bbox,
        page=page,
        confidence=confidence,
        requires_review=confidence < review_threshold,
        validated_against=controlled.lookup_name,
        components=ConfidenceComponents(
            ocr_confidence=ocr_confidence,
            controlled_match_score=controlled.score,
            quality_score=quality.overall_quality,
        ),
        source_text=source_text,
        reason_codes=[f"controlled_match:{controlled.lookup_name}"],
    )


def extract_regex_value(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1 if match.lastindex else 0).strip(" :-")


def extract_student_name(text: str) -> str | None:
    clean = re.sub(r"\s+", " ", text).strip()
    patterns = [
        r"(?:student\s+name|name\s+of\s+the\s+student|name)[\s:.-]+([A-Z][A-Za-z .'-]{2,60})",
        r"certify\s+that\s+([A-Z][A-Za-z .'-]{2,60}?)(?:\s+has|\s+is|\s+was)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, re.I)
        if match:
            return match.group(1).strip(" .:-")
    return None


def _maybe_set(
    fields: dict[str, ExtractedField],
    field_name: str,
    value: str | None,
    label: LabelClass,
    bbox: list[float],
    page: int,
    ocr_confidence: float,
    quality: QualityReport,
    *,
    source_text: str,
    reason: str,
    review_threshold: float,
) -> None:
    if not value:
        return
    confidence = bounded(mean(ocr_confidence, quality.overall_quality, 0.74))
    if field_name in fields and fields[field_name].confidence >= confidence:
        return
    fields[field_name] = ExtractedField(
        value=value,
        label_class=label,
        bbox=bbox,
        page=page,
        confidence=confidence,
        requires_review=confidence < review_threshold,
        components=ConfidenceComponents(ocr_confidence=ocr_confidence, quality_score=quality.overall_quality),
        source_text=source_text,
        reason_codes=[reason],
    )


def mean(*values: float | Iterable[float]) -> float:
    if len(values) == 1 and not isinstance(values[0], (int, float)):
        flat = [float(v) for v in values[0]]  # type: ignore[arg-type]
    else:
        flat = [float(v) for v in values]  # type: ignore[arg-type]
    flat = [v for v in flat if not math.isnan(v)]
    return sum(flat) / len(flat) if flat else 0.0


def bounded(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
