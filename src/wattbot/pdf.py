"""Extract PDF text page by page while preserving citation locations."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

from wattbot.data import read_csv


def extract_pages(data_dir: Path, pdf_dir: Path, output: Path) -> dict[str, int]:
    metadata = {row["id"]: row for row in read_csv(data_dir / "metadata.csv")}
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"documents": 0, "pages": 0, "empty_pages": 0, "missing_pdfs": 0, "errors": 0}
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as out:
        for doc_id, meta in metadata.items():
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.is_file():
                counts["missing_pdfs"] += 1
                continue
            try:
                reader = PdfReader(str(pdf_path), strict=False)
                extracted = []
                for page_number, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    extracted.append(
                        json.dumps(
                            {
                                "doc_id": doc_id,
                                "title": meta.get("title", ""),
                                "url": meta.get("url", ""),
                                "page": page_number,
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                out.writelines(extracted)
                counts["documents"] += 1
                counts["pages"] += len(extracted)
                counts["empty_pages"] += sum(not json.loads(line)["text"] for line in extracted)
                print(f"parsed {doc_id}: {len(extracted)} pages", flush=True)
            except Exception as exc:  # One damaged PDF should not discard the rest.
                counts["errors"] += 1
                print(f"ERROR {doc_id}: {type(exc).__name__}: {exc}")
    temporary.replace(output)
    return counts

