"""Acquire exact arXiv versions from its sanctioned Cornell/Kaggle GCS feed.

References: https://info.arxiv.org/help/bulk_data.html and
https://www.kaggle.com/datasets/Cornell-University/arxiv .
This module makes no requests to arxiv.org or publisher websites.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from wattbot.data import read_csv


def mirror_object(source_url: str, resolved_version: str | None = None) -> str | None:
    parsed = urllib.parse.urlsplit(source_url)
    if parsed.hostname not in {"arxiv.org", "www.arxiv.org"}:
        return None
    match = re.fullmatch(r"/(?:pdf|abs)/(\d{4}\.\d{4,5}v\d+)(?:\.pdf)?", parsed.path)
    if match:
        identifier = match.group(1)
    else:
        original = re.fullmatch(r"/(?:pdf|abs)/(\d{4}\.\d{4,5})(?:\.pdf)?", parsed.path)
        if not original or not resolved_version or not re.fullmatch(re.escape(original.group(1)) + r"v\d+", resolved_version):
            raise ValueError(f"Requires an explicit arXiv version or recorded resolution: {source_url}")
        identifier = resolved_version
    return f"arxiv/arxiv/pdf/{identifier[:4]}/{identifier}.pdf"


def fetch_arxiv_mirror(data_dir: Path, pdf_dir: Path, limit: int | None = None, version_map: Path | None = None) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    report = {"cached": [], "downloaded": [], "missing_versions": [], "errors": []}
    attempted = 0
    versions = json.loads(version_map.read_text(encoding="utf-8")).get("versions", {}) if version_map else {}
    for row in read_csv(data_dir / "metadata.csv"):
        try:
            object_name = mirror_object(row["url"].strip(), versions.get(row["id"]))
        except ValueError as exc:
            report["errors"].append({"doc_id": row["id"], "error": str(exc)})
            continue
        if object_name is None:
            continue
        doc_id = row["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", doc_id):
            raise ValueError(f"Invalid document ID: {doc_id!r}")
        target = pdf_dir / f"{doc_id}.pdf"
        if target.is_file():
            with target.open("rb") as handle:
                valid_header = handle.read(5) == b"%PDF-"
            if not valid_header:
                report["errors"].append({"doc_id": doc_id, "error": "Invalid cached PDF header"})
            else:
                report["cached"].append(doc_id)
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        encoded = urllib.parse.quote(object_name, safe="")
        info_url = f"https://storage.googleapis.com/storage/v1/b/arxiv-dataset/o/{encoded}"
        temp_target = target.with_suffix(".pdf.part")
        try:
            request = urllib.request.Request(info_url, headers={"User-Agent": "wattbot-rag/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                info = json.load(response)
            size = int(info["size"])
            if not 0 < size <= 250 * 1024 * 1024:
                raise ValueError(f"Unexpected object size: {size}")
            generation = str(info["generation"])
            media_url = (
                "https://storage.googleapis.com/download/storage/v1/b/arxiv-dataset/o/"
                f"{encoded}?generation={generation}&alt=media"
            )
            md5, sha256 = hashlib.md5(), hashlib.sha256()
            total = 0
            with urllib.request.urlopen(media_url, timeout=90) as response, temp_target.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > size:
                        raise ValueError("Received more bytes than mirror metadata declares")
                    md5.update(block)
                    sha256.update(block)
                    handle.write(block)
            if total != size:
                raise ValueError(f"Size mismatch: expected {size}, got {total}")
            if base64.b64encode(md5.digest()).decode("ascii") != info.get("md5Hash"):
                raise ValueError("Mirror MD5 checksum mismatch")
            with temp_target.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError("Mirror object is not a PDF")
            temp_target.replace(target)
            record = {
                "doc_id": doc_id,
                "url": row["url"],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "recording_method": "sanctioned_cornell_gcs_resolved_unpinned" if doc_id in versions else "sanctioned_cornell_gcs_exact_version",
                "version_resolution": "latest_available_in_mirror_original_url_unpinned" if doc_id in versions else "matches_version_in_metadata_url",
                "retrieved_from": media_url,
                "mirror_object": object_name,
                "generation": generation,
                "bytes": total,
                "sha256": sha256.hexdigest(),
                "md5_base64": info["md5Hash"],
            }
            with (pdf_dir / "download_manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            report["downloaded"].append(doc_id)
            print(f"cached {doc_id}: exact mirror object {object_name}, {total} bytes", flush=True)
        except urllib.error.HTTPError as exc:
            category = "missing_versions" if exc.code == 404 else "errors"
            report[category].append({"doc_id": doc_id, "http_status": exc.code, "object": object_name})
            print(f"{category} {doc_id}: HTTP {exc.code}", flush=True)
        except Exception as exc:
            report["errors"].append({"doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"error {doc_id}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if temp_target.exists():
                temp_target.unlink()
        time.sleep(0.25)
    report["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    (pdf_dir / "mirror_acquisition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
