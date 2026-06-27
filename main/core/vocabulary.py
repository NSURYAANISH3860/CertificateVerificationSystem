from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from main.core.settings import LOOKUP_DIR


@dataclass(frozen=True, slots=True)
class ControlledMatch:
    field_name: str
    value: str
    score: float
    lookup_name: str


DEFAULT_LOOKUP_FILES = {
    "degree": "degrees.json",
    "branch": "branches.json",
    "semester": "semesters.json",
    "regulation": "regulations.json",
    "subject_code": "subject_codes.json",
}

SUBJECT_CODE_RE = re.compile(r"\b[A-Z]{1,4}\d{2,4}[A-Z]{0,3}\b")
HALL_TICKET_RE = re.compile(
    r"\b(?:HT\s*NO\.?|HALL\s*TICKET(?:\s*(?:NO\.?|NUMBER))?|REG(?:ISTRATION)?\s*(?:NO\.?|NUMBER))[\s:.-]*((?=[A-Z0-9/-]*\d)[A-Z0-9/-]{5,})\b",
    re.I,
)
CGPA_RE = re.compile(r"\b(?:CGPA|SGPA|GPA)[\s:.-]*([0-9](?:\.\d{1,2})?|10(?:\.0{1,2})?)\b", re.I)
DATE_RE = re.compile(r"\b(?:\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}|\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})\b")
MARK_RE = re.compile(r"\b(?:100|[0-9]{1,2})(?:\.\d{1,2})?\b")


class ControlledVocabulary:
    def __init__(self, lookup_dir: str | Path = LOOKUP_DIR) -> None:
        self.lookup_dir = Path(lookup_dir)
        self.values: dict[str, list[str]] = {}
        self.load()

    def load(self) -> None:
        self.values.clear()
        for field_name, filename in DEFAULT_LOOKUP_FILES.items():
            path = self.lookup_dir / filename
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    values = loaded.get("values", [])
                else:
                    values = loaded
                self.values[field_name] = sorted({str(item).strip() for item in values if str(item).strip()})
            else:
                self.values[field_name] = []

    def match(self, text: str, *, threshold: float = 88.0) -> ControlledMatch | None:
        candidate = normalize_for_lookup(text)
        if not candidate:
            return None
        best: ControlledMatch | None = None
        for field_name, values in self.values.items():
            if not values:
                continue
            direct = {normalize_for_lookup(value): value for value in values}
            if candidate in direct:
                return ControlledMatch(field_name, direct[candidate], 1.0, f"{field_name}_lookup_v1")
            match = process.extractOne(candidate, direct.keys(), scorer=fuzz.WRatio)
            if not match:
                continue
            matched_key, score, _ = match
            normalized_score = float(score) / 100.0
            if score >= threshold and (best is None or normalized_score > best.score):
                best = ControlledMatch(field_name, direct[matched_key], normalized_score, f"{field_name}_lookup_v1")
        if best:
            return best
        subject_match = SUBJECT_CODE_RE.search(text.upper())
        if subject_match:
            return ControlledMatch("subject_code", subject_match.group(0), 0.86, "subject_code_pattern_v1")
        return None


def normalize_for_lookup(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
