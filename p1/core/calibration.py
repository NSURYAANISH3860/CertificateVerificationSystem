from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "ocr_confidence",
    "token_classifier_confidence",
    "layout_template_confidence",
    "repetition_rate",
    "controlled_match_score",
    "table_cell_confidence",
    "quality_score",
]


def train_calibration_model(
    review_csv: str | Path,
    output_path: str | Path,
    *,
    method: str = "logistic",
) -> Path:
    data = pd.read_csv(review_csv)
    if "correct" not in data.columns:
        raise ValueError("Review CSV must include a boolean/integer 'correct' column.")
    x = data.reindex(columns=FEATURE_COLUMNS).fillna(0.0)
    y = data["correct"].astype(int)

    if method == "isotonic":
        base_score = x.mean(axis=1)
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(base_score, y)
    elif method == "logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(x, y)
    else:
        raise ValueError("method must be 'logistic' or 'isotonic'")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump({"method": method, "features": FEATURE_COLUMNS, "model": model}, fh)
    return output


def predict_calibrated_probability(model_path: str | Path, features: dict[str, float | None]) -> float:
    with Path(model_path).open("rb") as fh:
        payload = pickle.load(fh)
    values = pd.DataFrame([{key: features.get(key) or 0.0 for key in payload["features"]}])
    if payload["method"] == "isotonic":
        score = values.mean(axis=1)
        return float(payload["model"].predict(score)[0])
    return float(payload["model"].predict_proba(values)[0, 1])
