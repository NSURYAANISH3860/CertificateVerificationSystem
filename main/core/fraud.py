from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from main.core.schemas import (
    ExtractionOutput,
    ExtractedField,
    FraudCheckDetail,
    OcrBox,
    TemplateVersion,
    VerificationReport,
    VerificationStatus,
)
from main.core.baseline import TemplateCluster

logger = logging.getLogger(__name__)

# Regular expressions for serial numbers per version
SERIAL_PATTERNS = {
    TemplateVersion.V1: re.compile(r"^[A-Z0-9]{3,12}$"),
    TemplateVersion.V2: re.compile(r"^JNTUH-\d{6}$", re.I),
    TemplateVersion.V3: re.compile(r"^\d{4}/[A-Z]/\d{2,4}$"),
    TemplateVersion.V4: re.compile(r"^CN\d{10}$"),
}


def verify_certificate(
    image: Image.Image,
    boxes: list[OcrBox],
    fields: dict[str, ExtractedField],
    claimed_year: int,
    institution: str = "JNTUH",
    doc_path: Path | None = None,
    template_profile: list[TemplateCluster] | None = None,
) -> VerificationReport:
    """
    Main verification pipeline that performs year checks, template matching,
    visual feature extraction, layout checks, and local anomaly detection.
    """
    detailed_checks: dict[str, FraudCheckDetail] = {}
    reasons: list[str] = []
    risk_score = 0.0

    # Convert PIL Image to OpenCV grayscale & RGB arrays for visual analysis
    img_rgb = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = img_gray.shape

    # 1. Extract and Verify Year from OCR
    detected_year = extract_year_from_ocr(boxes)
    if detected_year:
        year_status = (detected_year == claimed_year)
        detailed_checks["year_match"] = FraudCheckDetail(
            name="Year Verification",
            status=year_status,
            value=f"Claimed: {claimed_year}, Detected: {detected_year}",
            description="Checks if the year extracted from OCR matches the claimed passing year.",
        )
        if not year_status:
            reasons.append("OCR_YEAR_MISMATCH")
            risk_score += 0.4
    else:
        detailed_checks["year_match"] = FraudCheckDetail(
            name="Year Verification",
            status=True,
            value=f"Claimed: {claimed_year}, Detected: None",
            description="No award or passing year was found in OCR text. Human verification recommended.",
        )

    # 2. Detect Visual/Layout Features (QR, Seal, Signature)
    qr_data = detect_qr_code(img_rgb)
    seal_bbox = detect_seal(img_rgb)
    sig_bbox = detect_signature(img_gray)

    # 3. Template Version Classification and Alignment Checks
    if institution.upper() == "JNTUH":
        predicted_version, template_conf = classify_template_version(
            claimed_year, qr_data, seal_bbox, sig_bbox
        )
        template_correct = is_year_in_template_range(claimed_year, predicted_version)
        detailed_checks["template_match"] = FraudCheckDetail(
            name="Template Style & Version Match",
            status=template_correct,
            value=f"Predicted: {predicted_version.value} (Confidence: {template_conf:.2f})",
            description="Checks if the visual style, logo, seal, and QR code placement correspond to the claimed year's template version.",
        )
        if not template_correct:
            reasons.append("STYLE_YEAR_MISMATCH")
            risk_score += 0.35
        if predicted_version == TemplateVersion.UNKNOWN:
            reasons.append("UNKNOWN_TEMPLATE")
            risk_score += 0.2
    else:
        # Non-JNTUH institution layout alignment check
        if template_profile:
            template_conf = calculate_template_match_confidence(boxes, template_profile)
            template_correct = (template_conf >= 0.75)
            predicted_version = TemplateVersion.V0 if template_correct else TemplateVersion.UNKNOWN
            detailed_checks["template_match"] = FraudCheckDetail(
                name="Template Style & Version Match",
                status=template_correct,
                value=f"Profile Match: {template_conf*100:.1f}%",
                description="Checks if the text layout and constant blocks match the dynamically trained template profile.",
            )
            if not template_correct:
                reasons.append("TEMPLATE_MISMATCH")
                risk_score += 0.35
        else:
            predicted_version = TemplateVersion.V0
            template_conf = 1.0
            detailed_checks["template_match"] = FraudCheckDetail(
                name="Template Style & Version Match",
                status=True,
                value="New profile registered",
                description="This is the first document of this template. Layout saved as a reference for future validation.",
            )

    # 4. QR Code Presence & Payload Validation
    has_qr = (qr_data is not None)
    expected_qr = predicted_version in {TemplateVersion.V3, TemplateVersion.V4}
    
    if expected_qr and not has_qr:
        detailed_checks["qr_code"] = FraudCheckDetail(
            name="QR Code Presence",
            status=False,
            value="Missing",
            description=f"Template version {predicted_version.value} expects a verification QR code, but none was detected.",
        )
        reasons.append("QR_CODE_MISSING")
        risk_score += 0.3
    elif has_qr:
        # Check payload
        qr_text = qr_data.get("text", "")
        payload_ok = validate_qr_payload(qr_text, fields)
        detailed_checks["qr_code"] = FraudCheckDetail(
            name="QR Code Payload Verification",
            status=payload_ok,
            value="Decoded successfully",
            description="Verifies if the embedded QR code verification payload matches the text printed on the certificate.",
        )
        if not payload_ok:
            reasons.append("QR_CODE_PAYLOAD_MISMATCH")
            risk_score += 0.5
    else:
        detailed_checks["qr_code"] = FraudCheckDetail(
            name="QR Code Presence",
            status=True,
            value="Not expected / None",
            description="No QR code is expected for this template version.",
        )

    # 5. Font & Alignment Tampering Detection
    font_consistency, font_desc = check_font_consistency(boxes, fields)
    detailed_checks["font_consistency"] = FraudCheckDetail(
        name="Font Family & Size Consistency",
        status=font_consistency,
        value="Consistent" if font_consistency else "Irregularities detected",
        description=font_desc,
    )
    if not font_consistency:
        reasons.append("FONT_STYLE_INCONSISTENCY")
        risk_score += 0.25

    alignment_ok, align_desc = check_alignment_integrity(boxes, fields, height)
    detailed_checks["alignment_integrity"] = FraudCheckDetail(
        name="Layout & Line Alignment Check",
        status=alignment_ok,
        value="Aligned" if alignment_ok else "Misaligned fields detected",
        description=align_desc,
    )
    if not alignment_ok:
        reasons.append("ALIGNMENT_ANOMALY")
        risk_score += 0.25

    # 6. Background Copy-Paste (Noise/Texture) Verification
    noise_ok, noise_desc = check_background_noise_anomaly(img_gray, fields)
    detailed_checks["background_noise"] = FraudCheckDetail(
        name="Local Background Texture Check",
        status=noise_ok,
        value="Clean background" if noise_ok else "Potential edit artifacts",
        description=noise_desc,
    )
    if not noise_ok:
        reasons.append("COPY_PASTE_SUSPECT")
        risk_score += 0.3

    # 7. Serial Number Format Check
    serial_ok, serial_desc = check_serial_number_format(fields, predicted_version)
    detailed_checks["serial_number"] = FraudCheckDetail(
        name="Serial Number Format Verification",
        status=serial_ok,
        value=fields.get("serial_number", fields.get("hall_ticket_number")).value if (fields.get("serial_number") or fields.get("hall_ticket_number")) else "None",
        description=serial_desc,
    )
    if not serial_ok:
        reasons.append("SERIAL_NUMBER_FORMAT_INVALID")
        risk_score += 0.2

    # 8. Metadata and digital signatures scanner
    meta_ok, meta_desc = check_pdf_metadata_forensics(doc_path, image)
    detailed_checks["metadata_forensics"] = FraudCheckDetail(
        name="Digital Metadata Scan",
        status=meta_ok,
        value="Clean metadata" if meta_ok else "Software trace detected",
        description=meta_desc,
    )
    if not meta_ok:
        reasons.append("EDIT_SOFTWARE_DETECTED")
        risk_score += 0.35

    # 9. Error Level Analysis (ELA) check
    ela_ok, ela_desc = check_ela_anomaly(image, fields)
    detailed_checks["ela_forensics"] = FraudCheckDetail(
        name="Error Level Analysis (ELA)",
        status=ela_ok,
        value="Consistent compression" if ela_ok else "Anomalous compression",
        description=ela_desc,
    )
    if not ela_ok:
        reasons.append("ELA_ANOMALY")
        risk_score += 0.3

    # 10. JNTUH Cohort Verification
    if institution == "JNTUH":
        cohort_ok, cohort_desc = check_hall_ticket_year_consistency(fields, claimed_year)
        detailed_checks["hall_ticket_correlation"] = FraudCheckDetail(
            name="Hall Ticket Cohort Check",
            status=cohort_ok,
            value="Valid cohort correlation" if cohort_ok else "Cohort year anomaly",
            description=cohort_desc,
        )
        if not cohort_ok:
            reasons.append("COHORT_YEAR_MISMATCH")
            risk_score += 0.40

    # Final decision routing
    risk_score = min(1.0, max(0.0, risk_score))
    if risk_score >= 0.6:
        status = VerificationStatus.FLAGGED
    elif risk_score >= 0.2 or predicted_version == TemplateVersion.UNKNOWN:
        status = VerificationStatus.HUMAN_REVIEW
    else:
        status = VerificationStatus.VALID

    return VerificationReport(
        status=status,
        risk_score=round(risk_score, 3),
        reasons=reasons,
        claimed_year=claimed_year,
        detected_year=detected_year,
        predicted_template_version=predicted_version,
        template_match_confidence=round(template_conf, 2),
        detailed_checks=detailed_checks,
    )


def extract_year_from_ocr(boxes: list[OcrBox]) -> int | None:
    """
    Search for four-digit years in the certificate text and return the one
    associated with degree completion or award.
    """
    year_re = re.compile(r"\b((?:19|20)\d{2})\b")
    candidates: list[tuple[int, int]] = []  # (year, score)

    for box in boxes:
        text = box.text
        match = year_re.search(text)
        if match:
            year = int(match.group(1))
            if 1980 <= year <= 2026:
                # Calculate a confidence score for the year based on adjacent keywords
                score = 1
                lower_txt = text.lower()
                keywords = ["passed", "held in", "convocation", "award", "date", "year", "completion", "examination"]
                for kw in keywords:
                    if kw in lower_txt:
                        score += 3
                candidates.append((year, score))

    if not candidates:
        return None
    # Return the candidate year with the highest keyword matching score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_qr_code(img_rgb: np.ndarray) -> dict[str, Any] | None:
    """
    Uses OpenCV's QRCodeDetector to locate and decode any QR code.
    """
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img_rgb)
    if retval:
        for text, pts in zip(decoded_info, points):
            if text and pts is not None and len(pts) >= 4:
                h, w = img_rgb.shape[:2]
                x0 = float(pts[:, 0].min()) / w
                y0 = float(pts[:, 1].min()) / h
                x1 = float(pts[:, 0].max()) / w
                y1 = float(pts[:, 1].max()) / h
                return {
                    "text": text,
                    "normalized_bbox": [x0, y0, x1, y1],
                }
    return None


def detect_seal(img_rgb: np.ndarray) -> list[float] | None:
    """
    Detect colored seals/stamps (e.g. blue, red, purple ink) using HSV color segmentation.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Define ranges for typical stamp inks (reds, blues, purples)
    lower_blue = np.array([90, 40, 50])
    upper_blue = np.array([135, 255, 245])
    
    lower_red1 = np.array([0, 40, 50])
    upper_red1 = np.array([10, 255, 245])
    lower_red2 = np.array([165, 40, 50])
    upper_red2 = np.array([180, 255, 245])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask = cv2.bitwise_or(mask_blue, mask_red)

    # Find large contours representing seals
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_rgb.shape[:2]
    large_contours = []
    
    for c in contours:
        area = cv2.contourArea(c)
        # Filters out tiny noise spots and overly large backgrounds
        if 800 < area < 80000:
            x, y, cw, ch = cv2.boundingRect(c)
            # Check aspect ratio (seals are roughly square/circle-like, aspect ratio 0.5 to 2.0)
            aspect_ratio = float(cw) / ch
            if 0.4 <= aspect_ratio <= 2.5:
                large_contours.append((c, area, [float(x)/w, float(y)/h, float(x+cw)/w, float(y+ch)/h]))

    if not large_contours:
        return None

    # Return bounding box of the largest valid seal candidate
    large_contours.sort(key=lambda x: x[1], reverse=True)
    return large_contours[0][2]


def detect_signature(img_gray: np.ndarray) -> list[float] | None:
    """
    Heuristic to locate signature strokes in the bottom-right quadrant of the document.
    """
    h, w = img_gray.shape
    # Focus on lower right quadrant (y > 0.70, x > 0.60)
    quadrant = img_gray[int(h * 0.70):, int(w * 0.60):]
    
    # Adaptive thresholding to isolate strokes
    thresh = cv2.adaptiveThreshold(
        quadrant, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    
    # Find contours representing strokes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if 150 < area < 10000:
            x, y, cw, ch = cv2.boundingRect(c)
            # Map coordinates back to full image space
            fx0 = (int(w * 0.60) + x) / w
            fy0 = (int(h * 0.70) + y) / h
            fx1 = (int(w * 0.60) + x + cw) / w
            fy1 = (int(h * 0.70) + y + ch) / h
            sig_boxes.append((area, [fx0, fy0, fx1, fy1]))

    if not sig_boxes:
        return None
    # Merge nearby signature stroke contours
    min_x = min(box[1][0] for box in sig_boxes)
    min_y = min(box[1][1] for box in sig_boxes)
    max_x = max(box[1][2] for box in sig_boxes)
    max_y = max(box[1][3] for box in sig_boxes)
    return [min_x, min_y, max_x, max_y]


def classify_template_version(
    claimed_year: int,
    qr_data: dict[str, Any] | None,
    seal_bbox: list[float] | None,
    sig_bbox: list[float] | None,
) -> tuple[TemplateVersion, float]:
    """
    Classifies the template version based on the presence and layout
    of QR code, Seal and Signatures. Returns predicted version and confidence.
    """
    has_qr = (qr_data is not None)
    qr_y = qr_data["normalized_bbox"][1] if has_qr else 1.0
    qr_x = qr_data["normalized_bbox"][0] if has_qr else 1.0
    
    seal_y = seal_bbox[1] if seal_bbox else 0.0
    seal_x = seal_bbox[0] if seal_bbox else 0.0

    scores = {}

    # Score V0: 1980-2013 (No QR, classic layouts)
    v0_score = 0.0
    if not has_qr:
        v0_score += 0.4
    scores[TemplateVersion.V0] = v0_score

    # Score V1: 2014-2016 (No QR, Seal bottom-left y > 0.7, x < 0.4)
    v1_score = 0.0
    if not has_qr:
        v1_score += 0.5
    if seal_bbox and seal_y > 0.7 and seal_x < 0.4:
        v1_score += 0.5
    scores[TemplateVersion.V1] = v1_score

    # Score V2: 2017-2019 (No QR, Seal middle-left 0.4 <= y <= 0.7, x < 0.4)
    v2_score = 0.0
    if not has_qr:
        v2_score += 0.5
    if seal_bbox and 0.4 <= seal_y <= 0.75 and seal_x < 0.4:
        v2_score += 0.5
    scores[TemplateVersion.V2] = v2_score

    # Score V3: 2020-2022 (QR top-right y < 0.25, x > 0.6, Seal bottom-left y > 0.7, x < 0.4)
    v3_score = 0.0
    if has_qr and qr_y < 0.25 and qr_x > 0.6:
        v3_score += 0.5
    if seal_bbox and seal_y > 0.7 and seal_x < 0.4:
        v3_score += 0.5
    scores[TemplateVersion.V3] = v3_score

    # Score V4: 2023-2025 (QR bottom-left y > 0.7, x < 0.4, Seal bottom-center y > 0.7, 0.35 <= x <= 0.65)
    v4_score = 0.0
    if has_qr and qr_y > 0.7 and qr_x < 0.4:
        v4_score += 0.5
    if seal_bbox and seal_y > 0.7 and 0.3 <= seal_x <= 0.7:
        v4_score += 0.5
    scores[TemplateVersion.V4] = v4_score

    # Select best version
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_ver, best_score = sorted_scores[0]
    
    # If no landmarks are detected, default to year-based classification but with low confidence
    if best_score < 0.4:
        if 1980 <= claimed_year <= 2013:
            return TemplateVersion.V0, 0.3
        elif 2014 <= claimed_year <= 2016:
            return TemplateVersion.V1, 0.3
        elif 2017 <= claimed_year <= 2019:
            return TemplateVersion.V2, 0.3
        elif 2020 <= claimed_year <= 2022:
            return TemplateVersion.V3, 0.3
        elif 2023 <= claimed_year <= 2026:
            return TemplateVersion.V4, 0.3
        return TemplateVersion.UNKNOWN, 0.0

    return best_ver, best_score


def is_year_in_template_range(year: int, version: TemplateVersion) -> bool:
    if version == TemplateVersion.V0:
        return 1980 <= year <= 2013
    if version == TemplateVersion.V1:
        return 2014 <= year <= 2016
    if version == TemplateVersion.V2:
        return 2017 <= year <= 2019
    if version == TemplateVersion.V3:
        return 2020 <= year <= 2022
    if version == TemplateVersion.V4:
        return 2023 <= year <= 2026
    return False


def validate_qr_payload(qr_text: str, fields: dict[str, ExtractedField]) -> bool:
    """
    Validates if the decoded QR code payload contains credentials matching
    the extracted printed variables (e.g. registration/hall ticket number, student name).
    """
    normalized_qr = qr_text.lower()
    checks = []
    
    # Check registration / hall ticket number
    ht = fields.get("hall_ticket_number")
    if ht and ht.value:
        ht_val = ht.value.lower().strip()
        checks.append(ht_val in normalized_qr)
        
    # Check student name
    name = fields.get("student_name")
    if name and name.value:
        # Check first word or clean name parts to prevent minor string mismatches
        parts = [p.lower() for p in name.value.split() if len(p) > 2]
        if parts:
            checks.append(any(part in normalized_qr for part in parts))

    # Check CGPA
    cgpa = fields.get("cgpa")
    if cgpa and cgpa.value:
        checks.append(cgpa.value in normalized_qr)

    # Return True if all performed checks pass (ignore if no variables are extracted)
    return all(checks) if checks else True


def check_font_consistency(boxes: list[OcrBox], fields: dict[str, ExtractedField]) -> tuple[bool, str]:
    """
    Compares font size (bounding box height-to-width ratio) of student name and CGPA
    against the baseline font structures of surrounding constant labels.
    """
    variable_heights = []
    constant_heights = []

    # Map variables to locate their boxes
    var_box_texts = []
    for var_name in ["student_name", "cgpa"]:
        field = fields.get(var_name)
        if field and field.value:
            var_box_texts.extend([w.lower() for w in field.value.split()])

    for box in boxes:
        text_lower = box.text.lower()
        h_px = box.bbox[3] - box.bbox[1]
        
        if any(v_txt in text_lower for v_txt in var_box_texts):
            variable_heights.append(h_px)
        else:
            # Accumulate heights of other regular words as baseline
            if len(box.text) > 3 and not box.text.isdigit():
                constant_heights.append(h_px)

    if not variable_heights or not constant_heights:
        return True, "Insufficient text elements to compare fonts."

    avg_var = sum(variable_heights) / len(variable_heights)
    avg_const = sum(constant_heights) / len(constant_heights)

    # If key variable names/marks are scaled/font-size shifted by more than 2.0x, flag it
    ratio = avg_var / avg_const
    if ratio > 2.2 or ratio < 0.4:
        return False, f"Font size ratio mismatch detected: variable fields average height is {avg_var:.1f}px vs constants {avg_const:.1f}px."

    return True, f"Fonts are consistent. Height ratio is {ratio:.2f}."


def check_alignment_integrity(
    boxes: list[OcrBox], fields: dict[str, ExtractedField], page_height: int
) -> tuple[bool, str]:
    """
    Verifies vertical alignment of text lines containing variables.
    If a variable is copy-pasted, it is often slightly shifted vertically
    relative to adjacent constant words on the same line.
    """
    # Group boxes into crude lines
    lines: dict[int, list[OcrBox]] = {}
    for box in boxes:
        y_center = (box.bbox[1] + box.bbox[3]) / 2.0
        # Use a tolerance of 12 pixels for line grouping
        grouped = False
        for cy, line_boxes in lines.items():
            if abs(y_center - cy) < 14:
                line_boxes.append(box)
                grouped = True
                break
        if not grouped:
            lines[int(y_center)] = [box]

    anomalous_fields = []
    
    # Check variables
    for var_name in ["student_name", "cgpa", "hall_ticket_number"]:
        field = fields.get(var_name)
        if not field or not field.value:
            continue
        
        var_words = [w.lower() for w in field.value.split()]
        
        # Locate the line this variable is on
        for cy, line_boxes in lines.items():
            line_txt = [b.text.lower() for b in line_boxes]
            has_var = any(any(v_w in lt for lt in line_txt) for v_w in var_words)
            if has_var and len(line_boxes) > 2:
                # Calculate alignment (y-variance) of standard constant words on this line
                const_y_centers = []
                var_y_centers = []
                for b in line_boxes:
                    yc = (b.bbox[1] + b.bbox[3]) / 2.0
                    if any(v_w in b.text.lower() for v_w in var_words):
                        var_y_centers.append(yc)
                    else:
                        const_y_centers.append(yc)
                
                if const_y_centers and var_y_centers:
                    avg_const_y = sum(const_y_centers) / len(const_y_centers)
                    avg_var_y = sum(var_y_centers) / len(var_y_centers)
                    deviation = abs(avg_const_y - avg_var_y)
                    # If deviation exceeds 1.5% of page height, it's misaligned
                    if deviation > (page_height * 0.015):
                        anomalous_fields.append(f"{var_name} (shifted by {deviation:.1f}px)")

    if anomalous_fields:
        return False, f"Misaligned fields: {', '.join(anomalous_fields)}."
    return True, "All fields are vertically aligned with their respective lines."


def check_background_noise_anomaly(
    img_gray: np.ndarray, fields: dict[str, ExtractedField]
) -> tuple[bool, str]:
    """
    Performs Laplacian variance check on key variable regions.
    Detects if a white block or a blurry crop was pasted to replace text,
    causing the texture/noise profile of that box to differ from the page.
    """
    h, w = img_gray.shape
    local_variances = []
    
    # Sample a baseline clean background region (e.g. top-left corner, 100x100 pixels)
    bg_patch = img_gray[10:110, 10:110]
    bg_std = float(np.std(bg_patch))

    anomalous_fields = []

    for var_name in ["student_name", "cgpa", "hall_ticket_number"]:
        field = fields.get(var_name)
        if not field or not field.value:
            continue
        
        # Get bounding box coordinates in pixels
        bbox = field.bbox
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Add 5px padding
        x0, y0 = max(0, x0 - 5), max(0, y0 - 5)
        x1, y1 = min(w, x1 + 5), min(h, y1 + 5)
        
        if x1 - x0 < 10 or y1 - y0 < 5:
            continue
            
        crop = img_gray[y0:y1, x0:x1]
        
        # Calculate texture noise (Laplacian variance)
        lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        crop_std = float(np.std(crop))
        
        # If standard deviation is extremely low (nearly zero), it's a painted solid white/black box (erasure)
        if crop_std < 1.5:
            anomalous_fields.append(f"{var_name} (zero texture, std={crop_std:.2f})")
        # If variance is highly anomalous compared to surrounding text blocks (or extremely blurry)
        elif lap_var < 5.0:
            anomalous_fields.append(f"{var_name} (blurry texture, lap_var={lap_var:.1f})")

    if anomalous_fields:
        return False, f"Background texture anomaly in: {', '.join(anomalous_fields)}."
    
    return True, "No local background texture anomalies detected."


def check_serial_number_format(
    fields: dict[str, ExtractedField], version: TemplateVersion
) -> tuple[bool, str]:
    """
    Validates if the serial number or hall ticket format matches the expected regular expression
    for the identified template version.
    """
    serial_field = fields.get("serial_number") or fields.get("hall_ticket_number")
    if not serial_field or not serial_field.value:
        return True, "No serial number extracted to validate."

    val = serial_field.value.strip()
    pattern = SERIAL_PATTERNS.get(version)
    if not pattern:
        return True, f"No serial format pattern defined for version {version.value}."

    if not pattern.match(val):
        return False, f"Value '{val}' does not match the expected pattern for {version.value}."

    return True, f"Serial number format matches {version.value} template specification."


def check_pdf_metadata_forensics(doc_path: Path | None, image: Image.Image) -> tuple[bool, str]:
    """
    Scans document metadata or binary structure to detect editing software signatures.
    If doc_path is a PDF, scans its binary content for tool signatures.
    If it is an image, inspects PIL Image metadata tags.
    """
    editing_tools = [
        "photoshop", "illustrator", "canva", "acrobat pro", "nitro pdf", 
        "pdfescape", "inkscape", "gimp", "coreldraw", "indesign"
    ]
    
    # 1. Check Image EXIF/Metadata
    img_info = image.info or {}
    for key, val in img_info.items():
        if isinstance(val, str):
            val_lower = val.lower()
            for tool in editing_tools:
                if tool in val_lower:
                    return False, f"Image metadata indicates editing tool: {tool.title()} (field '{key}')."
                    
    # 2. Check PDF binary signatures
    if doc_path and doc_path.exists():
        if doc_path.suffix.lower() == ".pdf":
            try:
                file_size = doc_path.stat().st_size
                with open(doc_path, "rb") as f:
                    if file_size <= 32768:
                        content = f.read().lower()
                    else:
                        start_chunk = f.read(16384)
                        f.seek(file_size - 16384)
                        end_chunk = f.read(16384)
                        content = start_chunk + end_chunk
                
                content_str = content.decode("ascii", errors="ignore")
                
                found_tools = []
                for tool in editing_tools:
                    if tool in content_str:
                        found_tools.append(tool.title())
                
                if found_tools:
                    return False, f"PDF binary metadata indicates editing software: {', '.join(found_tools)}."
            except Exception as e:
                logger.warning(f"Error scanning PDF metadata: {e}")
                
    return True, "No digital editing signatures found in metadata."


def check_ela_anomaly(image: Image.Image, fields: dict[str, ExtractedField]) -> tuple[bool, str]:
    """
    Error Level Analysis (ELA) check.
    Saves the image as a JPEG at 90% quality, computes the difference, and checks
    if local variables have anomalous compression noise variances compared to a baseline.
    """
    from PIL import ImageChops
    import tempfile
    
    img_rgb = image.convert("RGB")
    width, height = img_rgb.size
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_name = Path(tmp.name)
    try:
        img_rgb.save(tmp_name, "JPEG", quality=90)
        compressed = Image.open(tmp_name)
        ela_diff = ImageChops.difference(img_rgb, compressed)
        ela_diff.load()
    finally:
        tmp_name.unlink(missing_ok=True)
        
    ela_gray = np.array(ela_diff.convert("L"))
    global_mean = float(np.mean(ela_gray))
    global_std = float(np.std(ela_gray))
    
    anomalous_fields = []
    
    for var_name in ["student_name", "cgpa", "hall_ticket_number"]:
        field = fields.get(var_name)
        if not field or not field.value:
            continue
            
        bbox = field.bbox
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x0, y0 = max(0, x0 - 3), max(0, y0 - 3)
        x1, y1 = min(width, x1 + 3), min(height, y1 + 3)
        
        if x1 - x0 < 5 or y1 - y0 < 3:
            continue
            
        crop_ela = ela_gray[y0:y1, x0:x1]
        crop_mean = float(np.mean(crop_ela))
        crop_std = float(np.std(crop_ela))
        
        if crop_std < 1.0:
            anomalous_fields.append(f"{var_name} (zero ELA noise, std={crop_std:.2f})")
        elif crop_mean > max(8.0, global_mean * 3.5):
            anomalous_fields.append(f"{var_name} (high ELA activity, mean={crop_mean:.1f} vs global={global_mean:.1f})")
            
    if anomalous_fields:
        return False, f"ELA compression anomaly detected in: {', '.join(anomalous_fields)}."
        
    return True, "Error Level Analysis (ELA) suggests consistent compression across variables."


def check_hall_ticket_year_consistency(fields: dict[str, ExtractedField], claimed_year: int) -> tuple[bool, str]:
    """
    Correlates JNTUH Hall Ticket / Registration number format with graduation year.
    First 2 digits represent the year of admission (e.g. '18' for 2018).
    Admission year must be consistent with graduation/claimed year (typically graduation is 3-6 years after admission).
    """
    ht_field = fields.get("hall_ticket_number") or fields.get("serial_number")
    if not ht_field or not ht_field.value:
        return True, "No hall ticket / registration number extracted to validate."
        
    val = ht_field.value.strip()
    match = re.match(r"^(\d{2})", val)
    if not match:
        return True, "Hall ticket format does not start with a 2-digit admission cohort year."
        
    yy = int(match.group(1))
    
    if claimed_year >= 2000:
        if yy > 70:  # Admitted in 19xx
            admission_year = 1900 + yy
        else:
            admission_year = 2000 + yy
    else:
        admission_year = 1900 + yy
        
    min_grad = admission_year + 2
    max_grad = admission_year + 7
    
    if not (min_grad <= claimed_year <= max_grad):
        return False, f"Cohort anomaly: Hall ticket '{val}' implies admission in {admission_year}, which is inconsistent with graduation in {claimed_year}."
        
    return True, f"Hall ticket year {admission_year} matches graduation cohort {claimed_year}."


def calculate_template_match_confidence(boxes: list[OcrBox], template_profile: list[TemplateCluster]) -> float:
    """
    Computes the percentage of constant text clusters in the template profile
    that are found in the current document boxes at their expected page and grid positions.
    """
    if not template_profile:
        return 1.0
        
    from main.core.baseline import position_bucket, normalize_text
    
    # Index document boxes by bucket
    doc_buckets = {}
    for box in boxes:
        bucket = position_bucket(box.normalized_bbox)
        doc_buckets[(box.page, bucket)] = normalize_text(box.text)
        
    matched = 0
    for cluster in template_profile:
        expected = normalize_text(cluster.canonical_text)
        actual = doc_buckets.get((cluster.page, cluster.bucket), "")
        if expected and actual and (expected in actual or actual in expected):
            matched += 1
            
    return float(matched / len(template_profile))

