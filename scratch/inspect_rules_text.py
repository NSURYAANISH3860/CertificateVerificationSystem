import fitz
doc = fitz.open("rules.pdf")
print(f"Number of pages: {len(doc)}")
for i, page in enumerate(doc):
    print(f"--- Page {i+1} ---")
    print(page.get_text()[:1000])
