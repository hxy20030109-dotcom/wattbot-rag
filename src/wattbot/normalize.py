"""Deterministic formatting for prediction CSVs before benchmark scoring."""

from __future__ import annotations

import csv
from pathlib import Path


_BOOLEAN_VALUES = {"true": "1", "false": "0"}


def normalize_answer_value(value: object) -> str:
    """Map boolean words to the benchmark's numeric labels and remove numeric commas."""
    normalized = str(value).strip()
    boolean = _BOOLEAN_VALUES.get(normalized.casefold())
    if boolean is not None:
        return boolean
    if normalized and normalized.casefold() not in {"is_blank", "nan", "none"}:
        try:
            float(normalized.replace(",", ""))
        except ValueError:
            return normalized
        if "," in normalized:
            return normalized.replace(",", "")
    return normalized


def normalize_predictions(input_path: Path, output_path: Path) -> dict:
    """Copy a CSV while canonicalizing only answer_value; refuse accidental overwrite."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    changed = []
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "answer_value" not in reader.fieldnames:
            raise ValueError("Input CSV must contain an answer_value column")
        with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                before = row["answer_value"]
                after = normalize_answer_value(before)
                if before != after:
                    changed.append({"id": row.get("id", ""), "before": before, "after": after})
                    row["answer_value"] = after
                writer.writerow(row)
    return {"rows_changed": len(changed), "changes": changed, "output": str(output_path.resolve())}
