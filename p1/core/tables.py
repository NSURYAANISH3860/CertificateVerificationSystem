from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from p1.core.baseline import group_boxes_into_lines, mean
from p1.core.io import merge_bboxes
from p1.core.schemas import ExtractedTable, OcrBox, TableCell
from p1.core.vocabulary import SUBJECT_CODE_RE


GRADE_RE = re.compile(r"\b(?:O|A\+|A|B\+|B|C|D|E|F|P|S)\b")
NUMERIC_RE = re.compile(r"^(?:100|[0-9]{1,2})(?:\.\d{1,2})?$")
HEADER_ALIASES = {
    "subject_code": {"code", "subject code", "sub code"},
    "subject_name": {"subject", "subject name", "paper", "course"},
    "internal_marks": {"internal", "int"},
    "external_marks": {"external", "ext"},
    "total_marks": {"total", "marks"},
    "grade": {"grade", "result"},
    "credits": {"credits", "credit", "cr"},
}


def extract_marksheet_tables_with_ppstructure(pages: list[tuple[int, Image.Image]]) -> list[ExtractedTable]:
    if not pages:
        return []
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import torch  # noqa: F401
        from paddleocr import PPStructure
    except Exception as exc:
        raise RuntimeError(f"PP-Structure is unavailable: {exc}") from exc

    engine = PPStructure(show_log=False, lang="en", table=True, ocr=True, formula=False)
    tables: list[ExtractedTable] = []
    for page_number, image in pages:
        result = engine(np.array(image.convert("RGB")))
        for item in result or []:
            table = ppstructure_item_to_table(item, page_number=page_number)
            if table:
                tables.append(table)
    return tables


def extract_marksheet_tables(boxes: list[OcrBox]) -> list[ExtractedTable]:
    lines = group_boxes_into_lines(boxes, y_tolerance=0.015)
    if not lines:
        return []

    header_index = find_header_line(lines)
    if header_index is None:
        return []

    headers = infer_headers(" ".join(box.text for box in lines[header_index]))
    rows: list[dict[str, str]] = []
    cells: list[TableCell] = []
    table_boxes: list[list[float]] = [box.bbox for box in lines[header_index]]

    for line_index, line in enumerate(lines[header_index + 1 :], start=1):
        line_text = " ".join(box.text for box in line)
        if not SUBJECT_CODE_RE.search(line_text.upper()):
            if rows and not any(NUMERIC_RE.match(box.text) for box in line):
                break
            continue
        row = infer_subject_row(line)
        if not row:
            continue
        rows.append(row)
        table_boxes.extend(box.bbox for box in line)
        for col_idx, (key, value) in enumerate(row.items()):
            matched_boxes = [box for box in line if value and value in box.text]
            bbox = merge_bboxes([box.bbox for box in matched_boxes] or [box.bbox for box in line])
            cells.append(
                TableCell(
                    row_index=len(rows) - 1,
                    column_index=col_idx,
                    header=key,
                    value=value,
                    bbox=bbox,
                    page=line[0].page,
                    confidence=float(mean(box.confidence for box in matched_boxes) if matched_boxes else mean(box.confidence for box in line)),
                )
            )

    if not rows:
        return []

    return [
        ExtractedTable(
            table_type="subject_marks",
            page=lines[header_index][0].page,
            bbox=merge_bboxes(table_boxes),
            headers=headers,
            rows=rows,
            cells=cells,
            confidence=float(mean(cell.confidence for cell in cells) if cells else 0.0),
        )
    ]


def ppstructure_item_to_table(item: dict[str, Any], *, page_number: int) -> ExtractedTable | None:
    item_type = str(item.get("type", "")).lower()
    if item_type != "table":
        return None
    res = item.get("res") or {}
    html = res.get("html") or item.get("html")
    rows = html_to_rows(html) if html else []
    bbox = item.get("bbox")
    if bbox is not None and len(bbox) == 4:
        bbox = [float(value) for value in bbox]
    else:
        bbox = None
    headers = list(rows[0].keys()) if rows else []
    return ExtractedTable(
        table_type="subject_marks",
        page=page_number,
        bbox=bbox,
        headers=headers,
        rows=rows,
        cells=[],
        confidence=0.80 if rows else 0.55,
        extraction_method="paddle_ppstructure",
    )


def html_to_rows(html: str) -> list[dict[str, str]]:
    try:
        frames = pd.read_html(html)
    except Exception:
        return []
    if not frames:
        return []
    frame = frames[0].fillna("")
    frame.columns = [normalize_header(str(column)) for column in frame.columns]
    return [
        {str(key): str(value).strip() for key, value in row.items() if str(value).strip()}
        for row in frame.to_dict(orient="records")
    ]


def normalize_header(header: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")
    return normalized or "column"


def find_header_line(lines: list[list[OcrBox]]) -> int | None:
    for idx, line in enumerate(lines):
        text = " ".join(box.text.lower() for box in line)
        hits = sum(term in text for term in ["subject", "code", "marks", "grade", "credit", "total"])
        if hits >= 2:
            return idx
    return None


def infer_headers(header_text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", header_text.lower()).strip()
    headers: list[str] = []
    for canonical, aliases in HEADER_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            headers.append(canonical)
    return headers or ["subject_code", "subject_name", "marks", "grade", "credits"]


def infer_subject_row(line: list[OcrBox]) -> dict[str, str] | None:
    tokens = [box.text.strip() for box in line if box.text.strip()]
    text = " ".join(tokens)
    subject_match = SUBJECT_CODE_RE.search(text.upper())
    if not subject_match:
        return None
    subject_code = subject_match.group(0)
    numbers = [token for token in tokens if NUMERIC_RE.match(token)]
    grade = next((token for token in tokens if GRADE_RE.fullmatch(token.upper())), "")

    subject_name_tokens: list[str] = []
    after_code = False
    for token in tokens:
        if token.upper() == subject_code:
            after_code = True
            continue
        if after_code and not NUMERIC_RE.match(token) and not GRADE_RE.fullmatch(token.upper()):
            subject_name_tokens.append(token)

    row = {
        "subject_code": subject_code,
        "subject_name": " ".join(subject_name_tokens).strip(),
    }
    if len(numbers) >= 1:
        row["internal_marks"] = numbers[0]
    if len(numbers) >= 2:
        row["external_marks"] = numbers[1]
    if len(numbers) >= 3:
        row["total_marks"] = numbers[2]
    if grade:
        row["grade"] = grade
    if len(numbers) >= 4:
        row["credits"] = numbers[-1]
    return row
