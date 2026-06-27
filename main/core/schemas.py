from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    DEGREE_CERTIFICATE = "degree_certificate"
    MARKSHEET = "marksheet"
    UNKNOWN = "unknown"


class LabelClass(str, Enum):
    CONSTANT = "CONSTANT"
    CONTROLLED_VARIABLE = "CONTROLLED_VARIABLE"
    OPEN_VARIABLE = "OPEN_VARIABLE"
    TABLE_HEADER = "TABLE_HEADER"
    TABLE_CELL = "TABLE_CELL"
    LOGO = "LOGO"
    SEAL = "SEAL"
    SIGNATURE = "SIGNATURE"
    PHOTO = "PHOTO"
    BACKGROUND = "BACKGROUND"
    NOISE_ARTIFACT = "NOISE_ARTIFACT"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    NEEDS_CORRECTION = "needs_correction"


class VerificationStatus(str, Enum):
    VALID = "VALID"
    FLAGGED = "FLAGGED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class TemplateVersion(str, Enum):
    V1 = "V1"  # 2014-2016
    V2 = "V2"  # 2017-2019
    V3 = "V3"  # 2020-2022
    V4 = "V4"  # 2023-2025
    UNKNOWN = "UNKNOWN"


class BBoxModel(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, value: list[float] | tuple[float, float, float, float]) -> "BBoxModel":
        return cls(x0=float(value[0]), y0=float(value[1]), x1=float(value[2]), y1=float(value[3]))

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


class QualityReport(BaseModel):
    blur_score: float = Field(ge=0.0, le=1.0)
    skew_deg: float
    low_contrast: bool
    contrast_score: float = Field(ge=0.0, le=1.0)
    resolution_score: float = Field(ge=0.0, le=1.0)
    compression_score: float = Field(default=0.0, ge=0.0, le=1.0)
    shadow_score: float = Field(default=0.0, ge=0.0, le=1.0)
    glare_score: float = Field(default=0.0, ge=0.0, le=1.0)
    crop_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_quality: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class OcrBox(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    text: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    normalized_bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    engine: str
    block_id: int | None = None
    line_id: int | None = None
    word_id: int | None = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ConfidenceComponents(BaseModel):
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    token_classifier_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    layout_template_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    controlled_match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    table_cell_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class ClassifiedRegion(BaseModel):
    text: str
    label_class: LabelClass
    bbox: list[float] = Field(min_length=4, max_length=4)
    normalized_bbox: list[float] = Field(min_length=4, max_length=4)
    page: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_review: bool
    components: ConfidenceComponents
    field_hint: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class ExtractedField(BaseModel):
    value: str
    label_class: LabelClass
    bbox: list[float] = Field(min_length=4, max_length=4)
    page: int = Field(default=1, ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_review: bool
    validated_against: str | None = None
    components: ConfidenceComponents = Field(default_factory=ConfidenceComponents)
    source_text: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING


class TableCell(BaseModel):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    header: str | None = None
    value: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    page: int = Field(default=1, ge=1)
    label_class: LabelClass = LabelClass.TABLE_CELL
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedTable(BaseModel):
    table_type: str
    page: int = Field(default=1, ge=1)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_method: str = "ocr_line_grouping"


class FraudCheckDetail(BaseModel):
    name: str
    status: bool  # True if passed (no fraud detected), False if failed
    value: str | None = None
    description: str


class VerificationReport(BaseModel):
    status: VerificationStatus
    risk_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    claimed_year: int
    detected_year: int | None = None
    predicted_template_version: TemplateVersion
    template_match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detailed_checks: dict[str, FraudCheckDetail] = Field(default_factory=dict)


class ExtractionOutput(BaseModel):
    schema_version: str = "1.0"
    project: str = "P1_FIELD_EXTRACTION"
    document_id: str
    institution: str
    document_type: DocumentType
    template_id: str | None = None
    source_file: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    regions: list[ClassifiedRegion] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    quality: QualityReport
    ocr_engine: str
    warnings: list[str] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationReport | None = None


class AnnotationObject(BaseModel):
    field_name: str
    label_class: LabelClass
    text: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    document_id: str
    template_id: str | None = None
    annotator_id: str | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING
