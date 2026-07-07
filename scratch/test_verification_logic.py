import json
from pathlib import Path
from PIL import Image
import numpy as np

from main.core.schemas import OcrBox
from main.core.fraud import verify_certificate

file_path = "c:/Users/Hp/CertificateVerificationSystem/data/templates/BOARD_OF_SECONDARY_EDUCATION_TELANGANA_SSC_CERTIFICATE_2024/DOC_D70199CC38D5.json"

with open(file_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

boxes = [OcrBox(**b) for b in raw]
fields = {}
img = Image.new("RGB", (800, 1000), "white")

report = verify_certificate(img, boxes, fields, claimed_year=2024, institution="BOARD_OF_SECONDARY_EDUCATION_TELANGANA")
print("Claimed year:", report.claimed_year)
print("Detected year:", report.detected_year)
print("Year Match check status:", report.detailed_checks["year_match"].status)
print("Year Match check value:", report.detailed_checks["year_match"].value)
