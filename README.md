# CVS Academic Document Field Extraction

CVS is a privacy-aware academic document extraction MVP for classifying OCR/layout content into:

- `CONSTANT`
- `CONTROLLED_VARIABLE`
- `OPEN_VARIABLE`

It supports degree certificates and marksheet/transcript-style documents, exports versioned JSON, and includes a review UI plus training entrypoints for LayoutLMv3 and confidence calibration once annotated data is available.

## What Is Implemented

- Upload/ingest for PDF, JPEG, PNG, TIFF, scanned images, and phone photos.
- Preprocessing with quality scoring: blur, skew, contrast, resolution, glare/shadow/crop/compression risk.
- OCR engine selection:
  - `auto`: prefers PaddleOCR if installed and initialized.
  - `paddleocr`: primary target engine from the spec.
  - `tesseract`: installed fallback path.
- Template-aware baseline scaffolding:
  - normalized OCR boxes,
  - position buckets,
  - repetition-rate template clusters,
  - `CONSTANT`, `CONTROLLED_VARIABLE`, `OPEN_VARIABLE`, and table header/cell labels.
- Controlled vocabulary validation for degree, branch, semester, regulation, and subject code values.
- Marksheet table reconstruction fallback using OCR line grouping.
- Structured JSON output with `schema_version: "1.0"`.
- FastAPI endpoint and Streamlit review interface.
- Model download script for PaddleOCR, LayoutLMv3, and optional TrOCR.
- Fine-tuning script for LayoutLMv3 token classification once annotations are supplied.
- Calibration script for logistic/isotonic confidence calibration from human review outcomes.

## Install

Core runtime:

```powershell
python -m pip install -r requirements.txt
```

ML/OCR model extras:

```powershell
python -m pip install -r requirements-ml.txt
```

Download/cache models:

```powershell
python scripts/download_models.py --layoutlmv3 --ppstructure
```

Optional OCR benchmark model:

```powershell
python scripts/download_models.py --layoutlmv3 --trocr
```

## Run The Interface

```powershell
streamlit run main/ui/app.py
```

## Run The API

```powershell
uvicorn main.api.main:app --reload --port 8000
```

Then post to `POST /extract` with a `file`, `document_type`, `institution`, optional `template_id`, and `ocr_engine`.

## Run From CLI

```powershell
python scripts/run_pipeline.py rules.pdf --document-type degree_certificate --ocr-engine tesseract
```

Output JSON is written under `outputs/`.

## Build A Template Profile

After you have multiple documents from the same template version:

```powershell
python scripts/build_template_profile.py data/samples/*.pdf --template-id JNTUH_CERT_V1 --ocr-engine auto
```

Then pass `--template-id JNTUH_CERT_V1` during extraction to use repetition-rate `CONSTANT` detection.

## Annotation Format

Use `data/annotation_schema.example.json` as the per-page JSONL shape for later training. Each line should include:

- `image_path`
- `image_width`
- `image_height`
- `annotations[]` with `field_name`, `label_class`, `text`, `bbox`, `document_id`, `template_id`, `annotator_id`, and `review_status`

## Fine-Tune LayoutLMv3

After the dataset is supplied:

```powershell
python scripts/train_layoutlmv3.py --annotations data/annotations.jsonl --output-dir models/layoutlmv3-cvs
```

Labels include BIO versions of `CONSTANT`, `CONTROLLED_VARIABLE`, `OPEN_VARIABLE`, `TABLE_HEADER`, `TABLE_CELL`, `SEAL`, `SIGNATURE`, and `LOGO`.

## Train Confidence Calibration

After human review labels are collected:

```powershell
python scripts/train_calibrator.py --review-csv data/review_outcomes.csv --output models/calibration/logistic.pkl
```

The review CSV must contain a `correct` column and can include component scores such as `ocr_confidence`, `layout_template_confidence`, `repetition_rate`, `controlled_match_score`, `table_cell_confidence`, and `quality_score`.
