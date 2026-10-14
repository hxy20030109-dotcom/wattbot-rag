"""Audit local PDF hashes, ingestion coverage and explicit version provenance."""

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pdf_dir = root / "data/pdfs"
processed = root / "data/processed"
with (root / "data/competition/metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
    metadata = list(csv.DictReader(handle))
records = [json.loads(line) for line in (pdf_dir / "download_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
manifest = {row["doc_id"]: row for row in records}
pages, empty_pages, first_pages = Counter(), Counter(), {}
if (processed / "pages.jsonl").exists():
    with (processed / "pages.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pages[row["doc_id"]] += 1
            empty_pages[row["doc_id"]] += not bool(row["text"].strip())
            if row["page"] == 1:
                first_pages[row["doc_id"]] = row["text"]
report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "expected_documents": len(metadata),
          "files": [], "missing": [], "hash_mismatches": [], "missing_manifest": [],
          "unpinned_metadata_ids": [], "version_header_mismatches": []}
for meta in metadata:
    doc_id = meta["id"]
    path = pdf_dir / f"{doc_id}.pdf"
    if not path.is_file():
        report["missing"].append(doc_id)
        continue
    record = manifest.get(doc_id, {})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not record:
        report["missing_manifest"].append(doc_id)
    elif digest != record.get("sha256"):
        report["hash_mismatches"].append(doc_id)
    is_arxiv = "arxiv.org/" in meta["url"]
    unpinned = is_arxiv and not re.search(r"v\d+(?:\.pdf)?$", meta["url"])
    if unpinned:
        report["unpinned_metadata_ids"].append(doc_id)
    expected = re.search(r"\d{4}\.\d{4,5}v\d+", record.get("mirror_object", "") or meta["url"])
    expected = expected.group(0) if expected else None
    header_versions = re.findall(r"arXiv:\s*(\d{4}\.\d{4,5}v\d+)", first_pages.get(doc_id, ""), flags=re.I)
    if expected and header_versions and expected not in header_versions:
        report["version_header_mismatches"].append({"doc_id": doc_id, "expected": expected, "observed": header_versions})
    report["files"].append({"doc_id": doc_id, "bytes": path.stat().st_size, "sha256": digest,
                            "pages": pages[doc_id], "empty_pages": empty_pages[doc_id],
                            "acquisition_method": record.get("recording_method"),
                            "resolved_version": expected, "version_in_first_page": expected in header_versions if expected else None,
                            "original_url_unpinned": unpinned})
report["cached_documents"] = len(report["files"])
report["parsed_documents"] = sum(row["pages"] > 0 for row in report["files"])
report["total_bytes"] = sum(row["bytes"] for row in report["files"])
report["total_pages"] = sum(pages.values())
report["empty_pages"] = sum(empty_pages.values())
report["chunk_count"] = sum(1 for _ in (processed / "chunks.jsonl").open(encoding="utf-8")) if (processed / "chunks.jsonl").exists() else 0
processed.mkdir(parents=True, exist_ok=True)
output = processed / "corpus_audit.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
print(output)
if report["missing"] or report["hash_mismatches"] or report["missing_manifest"] or report["version_header_mismatches"]:
    raise SystemExit(1)
