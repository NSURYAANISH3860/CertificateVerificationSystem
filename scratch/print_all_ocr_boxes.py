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
    ocr_engine="auto",
    claimed_year=2021,
    save=False,
)

print("All boxes returned by PaddleOCR:")
for r in output.regions:
    # Print if it has any numbers
    import re
    if re.search(r"\d", r.text):
        print(f"Text: '{r.text}'  Confidence: {r.confidence:.2f}")
