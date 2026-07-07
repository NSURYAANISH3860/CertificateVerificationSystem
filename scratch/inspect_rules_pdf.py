import pypdf
import os
from pathlib import Path
from main.core.io import load_document_images

# Let's print out if rules.pdf is a PDF and has text
try:
    from pdf2image import convert_from_path
    images = convert_from_path('rules.pdf')
    print(f"rules.pdf has {len(images)} pages.")
except Exception as e:
    print(f"Error loading with pdf2image: {e}")
