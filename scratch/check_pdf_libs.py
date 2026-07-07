import sys
for lib in ['pypdf', 'PyPDF2', 'pdfplumber', 'fitz', 'pdf2image']:
    try:
        __import__(lib)
        print(f"{lib} is installed")
    except ImportError:
        print(f"{lib} is NOT installed")
