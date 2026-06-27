from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoModelForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

BIO_LABELS = [
    "O",
    "B-CONSTANT",
    "I-CONSTANT",
    "B-CONTROLLED_VARIABLE",
    "I-CONTROLLED_VARIABLE",
    "B-OPEN_VARIABLE",
    "I-OPEN_VARIABLE",
    "B-TABLE_HEADER",
    "I-TABLE_HEADER",
    "B-TABLE_CELL",
    "I-TABLE_CELL",
    "B-SEAL",
    "I-SEAL",
    "B-SIGNATURE",
    "I-SIGNATURE",
    "B-LOGO",
    "I-LOGO",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 token classifier from CVS annotations.")
    parser.add_argument("--annotations", type=Path, required=True, help="JSONL file with image_path and annotations per document/page.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/layoutlmv3-cvs"))
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    args = parser.parse_args()

    records = read_annotation_jsonl(args.annotations)
    if not records:
        raise SystemExit("No records found. Provide annotations after dataset collection.")

    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    dataset = Dataset.from_list(records)
    encoded = dataset.map(lambda batch: encode_batch(batch, processor), batched=True, remove_columns=dataset.column_names)
    split = encoded.train_test_split(test_size=0.2, seed=42) if len(encoded) > 4 else {"train": encoded, "test": encoded}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(BIO_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=split["train"], eval_dataset=split["test"])
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved fine-tuned LayoutLMv3 model to {args.output_dir}")


def read_annotation_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            image_path = raw.get("image_path")
            annotations = raw.get("annotations", [])
            if not image_path or not annotations:
                continue
            words: list[str] = []
            boxes: list[list[int]] = []
            labels: list[int] = []
            width = raw.get("image_width") or Image.open(image_path).width
            height = raw.get("image_height") or Image.open(image_path).height
            for ann in annotations:
                label_class = ann.get("label_class", "OPEN_VARIABLE")
                pieces = str(ann.get("text", "")).split()
                if not pieces:
                    continue
                bbox = scale_bbox_0_1000(ann["bbox"], width, height)
                for idx, piece in enumerate(pieces):
                    prefix = "B" if idx == 0 else "I"
                    label = f"{prefix}-{label_class}"
                    words.append(piece)
                    boxes.append(bbox)
                    labels.append(LABEL_TO_ID.get(label, 0))
            records.append({"image_path": image_path, "words": words, "boxes": boxes, "labels": labels})
    return records


def encode_batch(batch: dict[str, list[Any]], processor: LayoutLMv3Processor) -> dict[str, Any]:
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
    encoding = processor(
        images,
        batch["words"],
        boxes=batch["boxes"],
        word_labels=batch["labels"],
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    return {key: np.array(value) for key, value in encoding.items()}


def scale_bbox_0_1000(bbox: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [
        int(max(0, min(1000, x0 / width * 1000))),
        int(max(0, min(1000, y0 / height * 1000))),
        int(max(0, min(1000, x1 / width * 1000))),
        int(max(0, min(1000, y1 / height * 1000))),
    ]


if __name__ == "__main__":
    main()
