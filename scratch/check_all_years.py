import json
import re
from pathlib import Path

data_dir = Path("c:/Users/Hp/CertificateVerificationSystem/data/templates")
year_re = re.compile(r"\b((?:19|20)\d{2})\b")

for path in data_dir.glob("**/*.json"):
    try:
        with path.open("r", encoding="utf-8") as f:
            boxes = json.load(f)
        years = []
        for box in boxes:
            text = box.get("text", "")
            for m in year_re.finditer(text):
                years.append(m.group(1))
        print(f"File: {path.relative_to(data_dir)} -> Size: {path.stat().st_size} -> Years: {years}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
