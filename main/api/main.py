from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from main.core.pipeline import process_document
from main.core.schemas import DocumentType
from main.core.settings import DEFAULT_INSTITUTION

app = FastAPI(title="CVS Academic Field Extraction", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": "CVS_FIELD_EXTRACTION"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(DocumentType.UNKNOWN),
    institution: str = Form(DEFAULT_INSTITUTION),
    template_id: str | None = Form(None),
    ocr_engine: str = Form("auto"),
) -> JSONResponse:
    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cvs_upload_") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        output = process_document(
            tmp_path,
            document_type=document_type,
            institution=institution,
            template_id=template_id or None,
            ocr_engine=ocr_engine,
        )
        return JSONResponse(output.model_dump(mode="json"))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/verify")
async def verify(
    file: UploadFile = File(...),
    claimed_year: int = Form(...),
    institution: str = Form(DEFAULT_INSTITUTION),
    ocr_engine: str = Form("auto"),
) -> JSONResponse:
    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cvs_verify_") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        output = process_document(
            tmp_path,
            document_type=DocumentType.DEGREE_CERTIFICATE,
            institution=institution,
            ocr_engine=ocr_engine,
            claimed_year=claimed_year,
        )
        return JSONResponse(output.model_dump(mode="json"))
    finally:
        tmp_path.unlink(missing_ok=True)
