from __future__ import annotations

from PIL import Image
import numpy as np

from main.core.fraud import (
    extract_year_from_ocr,
    classify_template_version,
    is_year_in_template_range,
    validate_qr_payload,
    verify_certificate,
    check_pdf_metadata_forensics,
    check_ela_anomaly,
    check_hall_ticket_year_consistency,
)
from main.core.schemas import OcrBox, ExtractedField, LabelClass, TemplateVersion, VerificationStatus


def test_extract_year_from_ocr() -> None:
    boxes = [
        OcrBox(document_id="d1", page=1, text="JNTU Hyderabad", bbox=[0,0,10,10], normalized_bbox=[0,0,0.1,0.1], confidence=0.9, engine="tess"),
        OcrBox(document_id="d1", page=1, text="Degree awarded in 2018", bbox=[0,0,10,10], normalized_bbox=[0,0,0.1,0.1], confidence=0.9, engine="tess"),
        OcrBox(document_id="d1", page=1, text="Some random text 2025", bbox=[0,0,10,10], normalized_bbox=[0,0,0.1,0.1], confidence=0.9, engine="tess"),
    ]
    year = extract_year_from_ocr(boxes)
    assert year == 2018  # 2018 is preferred because it has the keyword "awarded in"


def test_classify_template_version() -> None:
    # Test V4 classification: bottom-left QR code and bottom-center seal
    qr_data = {
        "text": "VERIFICATION_URL",
        "normalized_bbox": [0.1, 0.8, 0.25, 0.95],
    }
    seal_bbox = [0.45, 0.8, 0.55, 0.9]
    sig_bbox = [0.8, 0.8, 0.95, 0.9]

    version, confidence = classify_template_version(2024, qr_data, seal_bbox, sig_bbox)
    assert version == TemplateVersion.V4
    assert confidence >= 0.8


def test_is_year_in_template_range() -> None:
    assert is_year_in_template_range(1995, TemplateVersion.V0) is True
    assert is_year_in_template_range(2012, TemplateVersion.V0) is True
    assert is_year_in_template_range(2015, TemplateVersion.V0) is False
    assert is_year_in_template_range(2015, TemplateVersion.V1) is True
    assert is_year_in_template_range(2024, TemplateVersion.V1) is False
    assert is_year_in_template_range(2018, TemplateVersion.V2) is True
    assert is_year_in_template_range(2021, TemplateVersion.V3) is True
    assert is_year_in_template_range(2025, TemplateVersion.V4) is True


def test_validate_qr_payload() -> None:
    fields = {
        "student_name": ExtractedField(value="Rahul Kumar", label_class=LabelClass.OPEN_VARIABLE, bbox=[0,0,10,10], confidence=0.9, requires_review=False),
        "hall_ticket_number": ExtractedField(value="18031A0512", label_class=LabelClass.OPEN_VARIABLE, bbox=[0,0,10,10], confidence=0.9, requires_review=False),
    }
    qr_text = "Name: Rahul Kumar, HT: 18031A0512, CGPA: 8.5"
    assert validate_qr_payload(qr_text, fields) is True

    bad_qr_text = "Name: Alice, HT: 1234567, CGPA: 9.0"
    assert validate_qr_payload(bad_qr_text, fields) is False


def test_verify_certificate_valid_flow() -> None:
    # Create a simple plain white image
    img = Image.fromarray(np.ones((800, 600, 3), dtype=np.uint8) * 255)
    boxes = [
        OcrBox(document_id="d1", page=1, text="RAHUL KUMAR", bbox=[100, 100, 200, 120], normalized_bbox=[0.16, 0.12, 0.33, 0.15], confidence=0.95, engine="tess"),
        OcrBox(document_id="d1", page=1, text="2015", bbox=[100, 150, 150, 170], normalized_bbox=[0.16, 0.18, 0.25, 0.21], confidence=0.95, engine="tess"),
    ]
    fields = {
        "student_name": ExtractedField(value="RAHUL KUMAR", label_class=LabelClass.OPEN_VARIABLE, bbox=[100, 100, 200, 120], confidence=0.9, requires_review=False),
    }
    report = verify_certificate(img, boxes, fields, claimed_year=2015)
    # The default mock won't have QR or Seal landmarks, so version prediction will default based on year.
    assert report.claimed_year == 2015
    assert report.detected_year == 2015
    assert report.status in [VerificationStatus.VALID, VerificationStatus.HUMAN_REVIEW, VerificationStatus.FLAGGED]


def test_check_pdf_metadata_forensics() -> None:
    img = Image.new("RGB", (100, 100), color="white")
    ok, desc = check_pdf_metadata_forensics(None, img)
    assert ok is True
    
    img_infected = Image.new("RGB", (100, 100), color="white")
    img_infected.info["software"] = "Adobe Photoshop CC 2019"
    ok, desc = check_pdf_metadata_forensics(None, img_infected)
    assert ok is False
    assert "Photoshop" in desc


def test_check_ela_anomaly() -> None:
    img = Image.new("RGB", (200, 200), color="white")
    fields = {
        "student_name": ExtractedField(value="RAHUL KUMAR", label_class=LabelClass.OPEN_VARIABLE, bbox=[10, 10, 100, 30], confidence=0.9, requires_review=False),
    }
    ok, desc = check_ela_anomaly(img, fields)
    assert ok is False
    assert "zero ELA noise" in desc


def test_check_hall_ticket_year_consistency() -> None:
    fields = {
        "hall_ticket_number": ExtractedField(value="18031A0512", label_class=LabelClass.OPEN_VARIABLE, bbox=[0, 0, 10, 10], confidence=0.9, requires_review=False)
    }
    ok, desc = check_hall_ticket_year_consistency(fields, claimed_year=2022)
    assert ok is True
    
    ok, desc = check_hall_ticket_year_consistency(fields, claimed_year=2015)
    assert ok is False
    assert "Cohort anomaly" in desc


def test_verify_certificate_custom_institution_flow() -> None:
    from main.core.baseline import TemplateCluster
    img = Image.fromarray(np.ones((800, 600, 3), dtype=np.uint8) * 255)
    boxes = [
        OcrBox(document_id="d1", page=1, text="CONSTANT TEXT", bbox=[100, 100, 200, 120], normalized_bbox=[0.16, 0.12, 0.33, 0.15], confidence=0.95, engine="tess"),
        OcrBox(document_id="d1", page=1, text="2026", bbox=[100, 150, 150, 170], normalized_bbox=[0.16, 0.18, 0.25, 0.21], confidence=0.95, engine="tess"),
    ]
    fields = {}
    
    # 1. No profile loaded (first document) -> should match successfully (new profile registered)
    report_no_profile = verify_certificate(img, boxes, fields, claimed_year=2026, institution="ANNA_UNIVERSITY")
    assert report_no_profile.status in [VerificationStatus.VALID, VerificationStatus.HUMAN_REVIEW, VerificationStatus.FLAGGED]
    assert report_no_profile.detailed_checks["template_match"].status is True
    assert "New profile registered" in report_no_profile.detailed_checks["template_match"].value
    
    # 2. Profile loaded and matches
    template_profile = [
        TemplateCluster(page=1, bucket=(8, 4), repetition_rate=1.0, canonical_text="CONSTANT TEXT", canonical_bbox=[0.16, 0.12, 0.33, 0.15], count=1, total_documents=1)
    ]
    report_matching = verify_certificate(img, boxes, fields, claimed_year=2026, institution="ANNA_UNIVERSITY", template_profile=template_profile)
    assert report_matching.detailed_checks["template_match"].status is True
    assert "Profile Match: 100.0%" in report_matching.detailed_checks["template_match"].value
    
    # 3. Profile loaded and mismatches
    bad_boxes = [
        OcrBox(document_id="d1", page=1, text="DIFFERENT TEXT", bbox=[100, 100, 200, 120], normalized_bbox=[0.16, 0.12, 0.33, 0.15], confidence=0.95, engine="tess"),
    ]
    report_mismatch = verify_certificate(img, bad_boxes, fields, claimed_year=2026, institution="ANNA_UNIVERSITY", template_profile=template_profile)
    assert report_mismatch.detailed_checks["template_match"].status is False
    assert "Profile Match: 0.0%" in report_mismatch.detailed_checks["template_match"].value
