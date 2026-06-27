from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from main.core.pipeline import process_document
from main.core.schemas import DocumentType, VerificationStatus, TemplateVersion
from main.core.settings import DEFAULT_INSTITUTION
from main.core.io import load_document_images

# Premium Dark Mode Theme with Glassmorphism and Neon accents
st.set_page_config(page_title="Academic Authenticity Command Center", layout="wide")

st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Title Background Gradient */
    .command-header {
        background: linear-gradient(90deg, #1f6feb 0%, #8957e5 50%, #da36aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 38px;
        font-weight: 900;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    /* Navigation Glass Card */
    .nav-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Glowing Badges */
    .status-badge {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.5px;
        animation: pulse 2s infinite;
    }
    .badge-genuine {
        background: linear-gradient(135deg, #1b4d3e 0%, #0d281e 100%);
        border: 2px solid #2ea44f;
        color: #56d364;
        box-shadow: 0 0 25px rgba(46, 164, 79, 0.45);
    }
    .badge-suspicious {
        background: linear-gradient(135deg, #5c2020 0%, #2e0a0a 100%);
        border: 2px solid #f85149;
        color: #ff7b72;
        box-shadow: 0 0 25px rgba(248, 81, 73, 0.45);
    }
    .badge-review {
        background: linear-gradient(135deg, #5c4320 0%, #2b1d0a 100%);
        border: 2px solid #d29922;
        color: #ecad28;
        box-shadow: 0 0 25px rgba(210, 153, 34, 0.45);
    }
    
    /* Config Panel Cards */
    .config-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    /* Custom progress bar overrides */
    .stProgress > div > div > div > div {
        background-color: #58a6ff;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown("<div class='command-header'>🛡️ CERTIFICATE AUTHENTICITY FORENSICS PORTAL</div>", unsafe_allow_html=True)
st.caption("Advanced multi-dimensional validation platform for credential integrity verification, template alignment checks, and text alteration detection.")

# Config / Parameter Options in Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Engine Configurations")
    ocr_engine = st.selectbox("OCR Engine Select", ["auto", "paddleocr", "tesseract", "mock"], index=0)
    institution = st.text_input("Institution Identifier", value=DEFAULT_INSTITUTION)
    
    st.divider()
    st.markdown("### 🔍 Verification Rules")
    rule_year = st.checkbox("Year Validation", value=True)
    rule_template = st.checkbox("Template Geometry Match", value=True)
    rule_seal = st.checkbox("HSV Colored Seal Scan", value=True)
    rule_signature = st.checkbox("Signature Contour Density", value=True)
    rule_font = st.checkbox("Font Aspect Ratio Check", value=True)
    rule_align = st.checkbox("Vertical Regression Alignment", value=True)
    rule_noise = st.checkbox("Laplacian Noise Tampering", value=True)
    rule_qr = st.checkbox("QR Code Payload Match", value=True)
    rule_serial = st.checkbox("Serial Regex Validation", value=True)
    
    st.divider()
    st.markdown("### 🎚️ Threshold Parameters")
    risk_sensitivity = st.slider("Risk Alert Sensitivity Limit", 0.1, 0.9, 0.6, 0.05)
    dpi_setting = st.slider("PDF Resolution DPI", 150, 300, 220, 10)

# Visual Drawing Routine
def draw_verification_overlays(image: Image.Image, output) -> Image.Image:
    import cv2
    import numpy as np
    
    img = np.array(image.convert("RGB"))
    h, w, _ = img.shape
    overlay = img.copy()
    
    # Draw OCR bounding boxes
    for region in output.regions:
        x0, y0, x1, y1 = region.bbox
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if region.label_class.value == "CONSTANT":
            color = (135, 206, 250)
            thickness = 1
        elif region.label_class.value == "CONTROLLED_VARIABLE":
            color = (255, 215, 0)
            thickness = 2
        else:
            color = (200, 200, 200)
            thickness = 1
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
        
    # Draw Fields and highlight anomalies
    for field_name, field in output.fields.items():
        if not field.bbox:
            continue
        x0, y0, x1, y1 = [int(coord) for coord in field.bbox]
        
        is_anomalous = False
        if output.verification:
            checks = output.verification.detailed_checks
            font_ok = checks.get("font_consistency").status if "font_consistency" in checks else True
            align_ok = checks.get("alignment_integrity").status if "alignment_integrity" in checks else True
            noise_ok = checks.get("background_noise").status if "background_noise" in checks else True
            
            if field_name in ["student_name", "cgpa", "hall_ticket_number"]:
                if (rule_font and not font_ok) or (rule_align and not align_ok) or (rule_noise and not noise_ok):
                    is_anomalous = True
                    
        color = (235, 59, 90) if is_anomalous else (241, 196, 15)
        thickness = 3 if is_anomalous else 2
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(overlay, field_name.upper(), (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Highlight Landmarks
    if output.verification:
        from main.core.fraud import detect_qr_code, detect_seal, detect_signature
        qr = detect_qr_code(img)
        seal = detect_seal(img)
        sig = detect_signature(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        
        if rule_qr and qr:
            nb = qr["normalized_bbox"]
            qx0, qy0, qx1, qy1 = int(nb[0]*w), int(nb[1]*h), int(nb[2]*w), int(nb[3]*h)
            cv2.rectangle(overlay, (qx0, qy0), (qx1, qy1), (235, 77, 75), 3)
            cv2.putText(overlay, "VERIFICATION QR", (qx0, max(15, qy0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 77, 75), 2)
            
        if rule_seal and seal:
            sx0, sy0, sx1, sy1 = int(seal[0]*w), int(seal[1]*h), int(seal[2]*w), int(seal[3]*h)
            cv2.rectangle(overlay, (sx0, sy0), (sx1, sy1), (46, 204, 113), 3)
            cv2.putText(overlay, "SEAL / STAMP", (sx0, max(15, sy0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 2)
            
        if rule_signature and sig:
            sigx0, sigy0, sigx1, sigy1 = int(sig[0]*w), int(sig[1]*h), int(sig[2]*w), int(sig[3]*h)
            cv2.rectangle(overlay, (sigx0, sigy0), (sigx1, sigy1), (52, 152, 219), 3)
            cv2.putText(overlay, "SIGNATURE BLOCK", (sigx0, max(15, sigy0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 152, 219), 2)

    alpha = 0.82
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return Image.fromarray(blended)


# Main Tab Navigation Layout
tabs = st.tabs(["📂 Active Verification Command Desk", "⛓️ Batch Verification Engine", "📊 Template Matrix Forensics", "📝 Human Sign-Off Desk"])

with tabs[0]:
    col_inputs, col_outputs = st.columns([1.0, 1.0])
    
    with col_inputs:
        st.subheader("Document Ingestion")
        uploaded = st.file_uploader("Upload scanned certificate or transcript (PDF, JPG, PNG, TIF)", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"], key="single_uploader")
        
        claimed_year = st.number_input("Claimed graduation year", min_value=1980, max_value=2026, value=2024, step=1, key="single_year")
        
        if uploaded:
            st.info(f"Loaded file: {uploaded.name} ({uploaded.size / (1024*1024):.2f} MB)")
            
    with col_outputs:
        st.subheader("Verification Outcomes")
        if uploaded:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cvs_ui_") as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            try:
                with st.spinner("Processing preprocessing steps, rendering layouts, running OCR, and verifying rules..."):
                    # Process document
                    output = process_document(
                        tmp_path,
                        document_type=DocumentType.DEGREE_CERTIFICATE,
                        institution=institution,
                        ocr_engine=ocr_engine,
                        claimed_year=int(claimed_year),
                        ocr_lang="eng",
                        save=False,
                    )
                    
                    pages = load_document_images(tmp_path, dpi=dpi_setting)
                    orig_image = pages[0].image if pages else None
                    
            except Exception as exc:
                st.error(f"Execution failed: {exc}")
                st.stop()
            finally:
                tmp_path.unlink(missing_ok=True)

            # Verification Banner rendering
            if output.verification:
                ver = output.verification
                
                # Apply custom alert limits adjusted by sensitivity slider
                adjusted_risk = ver.risk_score
                if adjusted_risk >= risk_sensitivity:
                    adjusted_status = VerificationStatus.FLAGGED
                elif adjusted_risk >= 0.2:
                    adjusted_status = VerificationStatus.HUMAN_REVIEW
                else:
                    adjusted_status = VerificationStatus.VALID

                if adjusted_status == VerificationStatus.VALID:
                    st.markdown(
                        f"<div class='status-badge badge-genuine'>🛡️ SECURE: VERIFIED GENUINE (Risk Score: {adjusted_risk:.2f})</div>",
                        unsafe_allow_html=True,
                    )
                elif adjusted_status == VerificationStatus.FLAGGED:
                    st.markdown(
                        f"<div class='status-badge badge-suspicious'>🚨 ALERT: SUSPICIOUS / TAMPERING DETECTED (Risk Score: {adjusted_risk:.2f})</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='status-badge badge-review'>⚠️ WARNING: REQUIRES MANUAL FORENSIC REVIEW (Risk Score: {adjusted_risk:.2f})</div>",
                        unsafe_allow_html=True,
                    )
                    
                # Risk progress bar
                st.write("**Risk score:**")
                st.progress(adjusted_risk)
                
                if ver.reasons:
                    st.write("**Flagged Indicators:**")
                    for reason in ver.reasons:
                        st.markdown(f"- 🔴 `{reason}`")
                else:
                    st.markdown("- ✅ `NO_CRITICAL_FRAUD_FLAGS`")
            else:
                st.warning("Authenticity checks did not return a verification report.")

    if uploaded and orig_image:
        st.divider()
        col_preview, col_report = st.columns([1.1, 0.9])
        
        with col_preview:
            st.subheader("Highlighted Visual Detections")
            annotated_img = draw_verification_overlays(orig_image, output)
            st.image(annotated_img, use_container_width=True, caption="Legends: Blue = Constants, Yellow = Variables, Red = QR codes, Green = Seals, Orange = Signatures")
            
        with col_report:
            st.subheader("Verification Checklist Status")
            if output.verification:
                ver = output.verification
                
                # Render rules checklist
                checklist_data = []
                for key, check in ver.detailed_checks.items():
                    # Filter check based on active toggles
                    if key == "year_match" and not rule_year: continue
                    if key == "template_match" and not rule_template: continue
                    if key == "qr_code" and not rule_qr: continue
                    if key == "font_consistency" and not rule_font: continue
                    if key == "alignment_integrity" and not rule_align: continue
                    if key == "background_noise" and not rule_noise: continue
                    if key == "serial_number" and not rule_serial: continue
                    
                    status_icon = "🟢 Passed" if check.status else "🔴 Failed"
                    checklist_data.append({
                        "Check Name": check.name,
                        "Status": status_icon,
                        "Extracted Information": check.value,
                        "Description": check.description
                    })
                
                st.table(pd.DataFrame(checklist_data))
                
                # Show PDF conversion metadata
                st.markdown(
                    f"""
                    <div class='config-card'>
                    <strong>Extraction Metadata:</strong><br>
                    • Document ID: <code>{output.document_id}</code><br>
                    • OCR Engine: <code>{output.ocr_engine}</code><br>
                    • Bounding Regions Evaluated: <code>{len(output.regions)}</code><br>
                    • Named variables parsed: <code>{len(output.fields)}</code><br>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tabs[1]:
    st.subheader("Batch Verification Command")
    st.caption("Drop multiple certificate files to run rapid batch validation rules.")
    
    batch_files = st.file_uploader("Upload batch files", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"], accept_multiple_files=True, key="batch_uploader")
    batch_year = st.number_input("Common claimed graduation year", min_value=1980, max_value=2026, value=2024, step=1, key="batch_year")
    
    if batch_files:
        if st.button("Execute Batch Analysis"):
            batch_results = []
            progress_bar = st.progress(0.0)
            
            for idx, file in enumerate(batch_files):
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cvs_batch_") as tmp:
                    tmp.write(file.read())
                    tmp_path = Path(tmp.name)
                
                try:
                    out = process_document(
                        tmp_path,
                        document_type=DocumentType.DEGREE_CERTIFICATE,
                        institution=institution,
                        ocr_engine=ocr_engine,
                        claimed_year=int(batch_year),
                        save=False,
                    )
                    
                    if out.verification:
                        v = out.verification
                        status_str = "🛡️ GENUINE" if v.status == VerificationStatus.VALID else ("🚨 SUSPICIOUS" if v.status == VerificationStatus.FLAGGED else "⚠️ NEEDS REVIEW")
                        risk_val = v.risk_score
                        reasons_str = ", ".join(v.reasons) if v.reasons else "Clean"
                    else:
                        status_str = "UNKNOWN"
                        risk_val = 0.0
                        reasons_str = "No evaluation report"
                        
                except Exception as e:
                    status_str = "CRASHED"
                    risk_val = 1.0
                    reasons_str = str(e)
                finally:
                    tmp_path.unlink(missing_ok=True)
                    
                batch_results.append({
                    "File Name": file.name,
                    "Decision Status": status_str,
                    "Risk Score": risk_val,
                    "Flag Indicators": reasons_str
                })
                progress_bar.progress((idx + 1) / len(batch_files))
            
            st.success("Batch verification complete!")
            st.dataframe(pd.DataFrame(batch_results), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Landmark Coordinates Matrix")
    st.write("Compare extracted landmark positions (normalized coordinate grids) with JNTUH templates version profiles.")
    
    st.markdown(
        """
        | Template Version | Graduation Years | QR Code Location | Seal / Stamp Location | Signature Position |
        | --- | --- | --- | --- | --- |
        | **V0** | 1980 - 2013 | None | Variable / Historical | Bottom-Right (y: 0.8, x: 0.8) |
        | **V1** | 2014 - 2016 | None | Bottom-Left (y: 0.8, x: 0.1) | Bottom-Right (y: 0.8, x: 0.8) |
        | **V2** | 2017 - 2019 | None | Middle-Left (y: 0.5, x: 0.1) | Bottom-Right (y: 0.8, x: 0.8) |
        | **V3** | 2020 - 2022 | Top-Right (y: 0.1, x: 0.85) | Bottom-Left (y: 0.8, x: 0.1) | Bottom-Right (y: 0.8, x: 0.8) |
        | **V4** | 2023 - 2026 | Bottom-Left (y: 0.8, x: 0.1) | Bottom-Center (y: 0.8, x: 0.5) | Bottom-Right (y: 0.8, x: 0.8) |
        """
    )
    
    if uploaded and output.verification:
        st.divider()
        st.markdown("#### Detected Landmarks on Active Document:")
        from main.core.fraud import detect_qr_code, detect_seal, detect_signature
        
        # Recalculate landmarks
        img_np = np.array(orig_image)
        det_qr = detect_qr_code(img_np)
        det_seal = detect_seal(img_np)
        det_sig = detect_signature(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY))
        
        col_lm1, col_lm2, col_lm3 = st.columns(3)
        with col_lm1:
            st.metric("QR Code Bounding Box", str([round(c, 2) for c in det_qr["normalized_bbox"]]) if det_qr else "Not Detected")
        with col_lm2:
            st.metric("Seal / Stamp Bounding Box", str([round(c, 2) for c in det_seal]) if det_seal else "Not Detected")
        with col_lm3:
            st.metric("Signature Bounding Box", str([round(c, 2) for c in det_sig]) if det_sig else "Not Detected")

with tabs[3]:
    st.subheader("Forensic Log Sign-Off Console")
    st.caption("Submit manual validation audits to log genuine/suspicious overrides.")
    
    if uploaded:
        op_name = st.text_input("Human Auditor Name", value="Chief Forensic Officer")
        decision = st.selectbox("Verification Sign-Off Decision", ["CONFIRM GENUINE - Seal Document", "CONFIRM TAMPERED - Flag System", "REQUEST RE-SCAN - Low Quality"])
        remarks = st.text_area("Audit Forensic Remarks")
        
        if st.button("Log Sign-Off to Audit Trail"):
            log_path = Path("outputs") / "audit_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_entry = {
                "timestamp": time.time(),
                "document_id": output.document_id,
                "operator": op_name,
                "decision": decision,
                "remarks": remarks,
                "system_status": output.verification.status.value if output.verification else "none",
                "risk_score": output.verification.risk_score if output.verification else 0.0,
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry) + "\n")
            st.success(f"Audit log signed successfully for ID: {output.document_id} by {op_name}!")
    else:
        st.info("Ingest a certificate to enable manual forensic sign-offs.")
