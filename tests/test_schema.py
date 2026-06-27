from main.core.schemas import DocumentType, ExtractionOutput, QualityReport


def test_output_schema_version_is_present() -> None:
    output = ExtractionOutput(
        document_id="DOC_TEST",
        institution="JNTUH",
        document_type=DocumentType.DEGREE_CERTIFICATE,
        quality=QualityReport(
            blur_score=0.1,
            skew_deg=0.0,
            low_contrast=False,
            contrast_score=0.8,
            resolution_score=0.8,
            overall_quality=0.8,
        ),
        ocr_engine="tesseract",
    )
    payload = output.model_dump(mode="json")
    assert payload["schema_version"] == "1.0"
    assert payload["project"] == "CVS_FIELD_EXTRACTION"
