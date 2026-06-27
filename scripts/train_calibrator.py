from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.calibration import train_calibration_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P1 confidence calibration from human review outcomes.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/calibration/logistic.pkl"))
    parser.add_argument("--method", choices=["logistic", "isotonic"], default="logistic")
    args = parser.parse_args()
    path = train_calibration_model(args.review_csv, args.output, method=args.method)
    print(f"Saved calibration model to {path}")


if __name__ == "__main__":
    main()
