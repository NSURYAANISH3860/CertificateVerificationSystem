from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.pipeline import process_document

def create_mock_certificate_img(path: str):
    # Create white image of typical certificate size
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)
    
    # Draw simple textual representations
    draw.text((100, 50), "JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY", fill="black")
    draw.text((100, 150), "STUDENT NAME: RAHUL KUMAR", fill="black")
    draw.text((100, 250), "DEGREE: Bachelor of Technology", fill="black")
    draw.text((100, 350), "CGPA: 8.45", fill="black")
    draw.text((100, 450), "AWARD YEAR: 2015", fill="black")
    draw.text((100, 550), "SERIAL NUMBER: S1234567", fill="black")
    
    # Draw a colored green seal/stamp at the bottom-left
    draw.ellipse([150, 800, 220, 870], fill=(46, 204, 113)) # Emerald green stamp
    
    img.save(path)


def main():
    path = "mock_cert.jpg"
    create_mock_certificate_img(path)
    print(f"Created mock certificate image at {path}")
    
    try:
        # Run process_document pipeline on the image
        output = process_document(
            path,
            document_type="degree_certificate",
            claimed_year=2015,
            ocr_engine="mock", # Force Mock OCR Engine
            save=False,
        )
        print("\n--- Pipeline Execution Success ---")
        print(f"Document ID: {output.document_id}")
        print(f"OCR Engine Used: {output.ocr_engine}")
        if output.verification:
            ver = output.verification
            print(f"Verification Status: {ver.status.value}")
            print(f"Authenticity Risk Score: {ver.risk_score}")
            print(f"Predicted Template Version: {ver.predicted_template_version.value}")
            print(f"Template Similarity Confidence: {ver.template_match_confidence}")
            print("\nDetailed Checklist:")
            for name, check in ver.detailed_checks.items():
                print(f" - {check.name}: {'PASSED' if check.status else 'FAILED'} (Value: {check.value})")
        else:
            print("Verification was not run.")
            
    except Exception as exc:
        print(f"Pipeline crashed: {exc}")
    finally:
        # Clean up mock file
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
