"""Page-aware character chunking for the first retrieval baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path


def chunk_pages(source: Path, target: Path, size: int = 1800, overlap: int = 250) -> int:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with source.open("r", encoding="utf-8") as src, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in src:
            page = json.loads(line)
            text = re.sub(r"\s+", " ", page.get("text", "")).strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                if end < len(text):
                    boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
                    if boundary > start + size // 2:
                        end = boundary + 1
                chunk_text = text[start:end].strip()
                if chunk_text:
                    row = {k: page[k] for k in ("doc_id", "title", "url", "page")}
                    row.update({"chunk_id": f"{page['doc_id']}-p{page['page']}-c{total}", "text": chunk_text})
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
                if end >= len(text):
                    break
                start = max(start + 1, end - overlap)
    return total

