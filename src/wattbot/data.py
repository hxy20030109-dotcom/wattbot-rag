"""CSV and citation helpers for the WattBot benchmark."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(next(csv.reader(handle), []))


def is_blank(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"", "na", "nan", "is_blank"}


def parse_refs(value: Any) -> list[str]:
    """Parse the scorer's single IDs or stringified ID lists."""
    raw = str(value or "").strip()
    if is_blank(raw):
        return []
    try:
        parsed = json.loads(raw.replace("'", '"')) if raw.startswith("[") else raw
    except json.JSONDecodeError:
        parsed = raw
    values = parsed if isinstance(parsed, list) else re.split(r"[,;|]", str(parsed))
    return [str(item).strip().casefold() for item in values if str(item).strip()]


def flag(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}

