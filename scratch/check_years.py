import json
import re

file_path = "c:/Users/Hp/CertificateVerificationSystem/data/templates/BOARD_OF_SECONDARY_EDUCATION_TELANGANA_SSC_CERTIFICATE_2024/DOC_D70199CC38D5.json"

with open(file_path, "r", encoding="utf-8") as f:
    boxes = json.load(f)

year_re = re.compile(r"\b((?:19|20)\d{2})\b")
all_years = []
for box in boxes:
    text = box["text"]
    for match in year_re.finditer(text):
        all_years.append((text, match.group(1)))

print("Extracted Years from OCR boxes:")
for txt, yr in all_years:
    print(f"Text: '{txt}' -> Year: {yr}")
