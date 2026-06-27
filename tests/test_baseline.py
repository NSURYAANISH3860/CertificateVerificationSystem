from main.core.baseline import extract_fields
from main.core.schemas import OcrBox, QualityReport
from main.core.vocabulary import ControlledVocabulary


def test_extracts_controlled_degree_and_open_cgpa() -> None:
    quality = QualityReport(
        blur_score=0.1,
        skew_deg=0.0,
        low_contrast=False,
        contrast_score=0.8,
        resolution_score=0.9,
        overall_quality=0.9,
    )
    boxes = [
        OcrBox(
            document_id="DOC_TEST",
            page=1,
            text="Bachelor of Technology",
            bbox=[10, 10, 200, 30],
            normalized_bbox=[0.01, 0.01, 0.2, 0.03],
            confidence=0.95,
            engine="test",
        ),
        OcrBox(
            document_id="DOC_TEST",
            page=1,
            text="CGPA: 8.72",
            bbox=[10, 40, 200, 60],
            normalized_bbox=[0.01, 0.04, 0.2, 0.06],
            confidence=0.93,
            engine="test",
        ),
    ]
    fields = extract_fields(boxes, [], vocabulary=ControlledVocabulary(), quality=quality)
    assert fields["degree"].value == "Bachelor of Technology"
    assert fields["degree"].label_class.value == "CONTROLLED_VARIABLE"
    assert fields["cgpa"].value == "8.72"


def test_regulation_text_is_not_hall_ticket_number() -> None:
    quality = QualityReport(
        blur_score=0.1,
        skew_deg=0.0,
        low_contrast=False,
        contrast_score=0.8,
        resolution_score=0.9,
        overall_quality=0.9,
    )
    boxes = [
        OcrBox(
            document_id="DOC_TEST",
            page=1,
            text="Build lookup lists for degree, branch, semester, regulation, subject code.",
            bbox=[10, 10, 500, 30],
            normalized_bbox=[0.01, 0.01, 0.5, 0.03],
            confidence=0.95,
            engine="test",
        )
    ]
    fields = extract_fields(boxes, [], vocabulary=ControlledVocabulary(), quality=quality)
    assert "hall_ticket_number" not in fields


def test_hall_ticket_label_without_numeric_identifier_is_ignored() -> None:
    quality = QualityReport(
        blur_score=0.1,
        skew_deg=0.0,
        low_contrast=False,
        contrast_score=0.8,
        resolution_score=0.9,
        overall_quality=0.9,
    )
    boxes = [
        OcrBox(
            document_id="DOC_TEST",
            page=1,
            text="Student name, hall ticket Pattern validation, OCR",
            bbox=[10, 10, 500, 30],
            normalized_bbox=[0.01, 0.01, 0.5, 0.03],
            confidence=0.95,
            engine="test",
        ),
        OcrBox(
            document_id="DOC_TEST",
            page=1,
            text="Hall Ticket No: 20A91A0501",
            bbox=[10, 40, 500, 60],
            normalized_bbox=[0.01, 0.04, 0.5, 0.06],
            confidence=0.95,
            engine="test",
        ),
    ]
    fields = extract_fields(boxes, [], vocabulary=ControlledVocabulary(), quality=quality)
    assert fields["hall_ticket_number"].value == "20A91A0501"
