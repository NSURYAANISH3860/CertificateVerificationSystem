import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.core.pipeline import process_document
from main.core.schemas import DocumentType

pdf_path = Path("C:/Users/Hp/.gemini/antigravity-ide/brain/0eca8aa2-589a-4d66-a516-da304449e412/media__1783430852109.pdf")
output = process_document(
    pdf_path,
    document_type=DocumentType.SSC_CERTIFICATE,
    institution="Board of Secondary Education Telangana",
    ocr_engine="tesseract",
    claimed_year=2021,
    save=False,
)

print("Document ID:", output.document_id)
print("Detected Year:", output.verification.detected_year if output.verification else "None")
if output.verification:
    print("Year status:", output.verification.detailed_checks["year_match"].status)
    print("Year value:", output.verification.detailed_checks["year_match"].value)

# Print all boxes that contain any 4-digit number
import re
year_re = re.compile(r"\b((?:19|20)\d{2})\b")
for r in output.regions:
    if year_re.search(r.text):
        print(f"Region: '{r.text}'")
