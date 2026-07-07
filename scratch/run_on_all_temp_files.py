import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.core.pipeline import process_document
from main.core.schemas import DocumentType

dir_path = Path("C:/Users/Hp/.gemini/antigravity-ide/brain/0eca8aa2-589a-4d66-a516-da304449e412")
files = list(dir_path.glob("**/*.pdf")) + list(dir_path.glob("**/*.png")) + list(dir_path.glob("**/*.jpg"))

for f in files:
    if ".system_generated" in str(f):
        continue
    print(f"\n--- Processing {f.name} (size: {f.stat().st_size}) ---")
    try:
        output = process_document(
            f,
            document_type=DocumentType.SSC_CERTIFICATE,
            institution="Board of Secondary Education Telangana",
            ocr_engine="auto",
            claimed_year=2021,
            save=False,
        )
        print("Detected Year:", output.verification.detected_year if output.verification else "None")
        if output.verification:
            print("Year value:", output.verification.detailed_checks["year_match"].value)
        # Print all years in the boxes
        import re
        year_re = re.compile(r"\b((?:19|20)\d{2})\b")
        years_found = [r.text for r in output.regions if year_re.search(r.text)]
        print("Years found in OCR:", years_found)
    except Exception as e:
        print("Error:", e)
