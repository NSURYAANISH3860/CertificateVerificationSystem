from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from p1.core.pipeline import process_document
from p1.core.schemas import DocumentType
from p1.core.settings import DEFAULT_INSTITUTION


st.set_page_config(page_title="P1 Field Extraction", layout="wide")

st.title("P1 Academic Document Field Extraction")
st.caption("CONSTANT, CONTROLLED_VARIABLE, and OPEN_VARIABLE extraction for degree certificates and marksheets.")

with st.sidebar:
    st.header("Run")
    document_type = st.selectbox(
        "Document type",
        [DocumentType.DEGREE_CERTIFICATE.value, DocumentType.MARKSHEET.value, DocumentType.UNKNOWN.value],
    )
    institution = st.text_input("Institution", value=DEFAULT_INSTITUTION)
    template_id = st.text_input("Template ID", value="")
    ocr_engine = st.selectbox("OCR engine", ["auto", "paddleocr", "tesseract"], index=0)
    st.divider()
    st.write("Low-confidence fields are flagged for review instead of being silently accepted.")

uploaded = st.file_uploader("Upload PDF, image, scan, or phone photo", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"])

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="p1_ui_") as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Running preprocessing, OCR, field classification, and JSON export..."):
            output = process_document(
                tmp_path,
                document_type=document_type,
                institution=institution,
                template_id=template_id or None,
                ocr_engine=ocr_engine,
            )
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        st.stop()
    finally:
        tmp_path.unlink(missing_ok=True)

    st.success(f"Processed {uploaded.name} with {output.ocr_engine}.")
    if output.warnings:
        for warning in output.warnings:
            st.warning(warning)

    metrics = st.columns(5)
    metrics[0].metric("Fields", len(output.fields))
    metrics[1].metric("Regions", len(output.regions))
    metrics[2].metric("Tables", len(output.tables))
    metrics[3].metric("Quality", f"{output.quality.overall_quality:.2f}")
    metrics[4].metric("Needs Review", sum(1 for field in output.fields.values() if field.requires_review))

    fields_df = pd.DataFrame(
        [
            {
                "field": name,
                "value": field.value,
                "label": field.label_class.value,
                "confidence": round(field.confidence, 3),
                "requires_review": field.requires_review,
                "validated_against": field.validated_against,
                "reason_codes": ", ".join(field.reason_codes),
            }
            for name, field in output.fields.items()
        ]
    )

    tab_fields, tab_tables, tab_regions, tab_json = st.tabs(["Fields", "Tables", "Regions", "JSON"])
    with tab_fields:
        if fields_df.empty:
            st.info("No named fields were extracted yet. Review OCR regions below and add annotations when data is ready.")
        else:
            st.data_editor(fields_df, use_container_width=True, hide_index=True)

    with tab_tables:
        if not output.tables:
            st.info("No marksheet table was detected.")
        for table in output.tables:
            st.subheader(table.table_type)
            st.dataframe(pd.DataFrame(table.rows), use_container_width=True)

    with tab_regions:
        regions_df = pd.DataFrame(
            [
                {
                    "page": region.page,
                    "text": region.text,
                    "label": region.label_class.value,
                    "confidence": round(region.confidence, 3),
                    "requires_review": region.requires_review,
                    "hint": region.field_hint,
                    "reason_codes": ", ".join(region.reason_codes),
                }
                for region in output.regions
            ]
        )
        st.dataframe(regions_df, use_container_width=True, hide_index=True)

    with tab_json:
        payload = output.model_dump(mode="json")
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2),
            file_name=f"{output.document_id}.json",
            mime="application/json",
        )
        st.json(payload)
else:
    st.info("Upload a degree certificate or marksheet to run the MVP pipeline.")
