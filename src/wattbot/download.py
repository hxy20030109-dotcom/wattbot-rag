"""Polite, explicit-ID downloader for individually selected source PDFs."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from wattbot.data import read_csv


def download_selected(data_dir: Path, pdf_dir: Path, doc_ids: list[str], delay: float = 4.0) -> None:
    if not doc_ids:
        raise ValueError("Select exactly one document ID with --id; there is no bulk default.")
    if len(doc_ids) != 1:
        raise ValueError(
            "Fetch one source per invocation. Kaggle's current rules prohibit bulk downloaders "
            "against arXiv and publisher sites; ask the organizers about a sanctioned corpus route."
        )
    metadata = {row["id"]: row for row in read_csv(data_dir / "metadata.csv")}
    unknown = sorted(set(doc_ids) - metadata.keys())
    if unknown:
        raise ValueError(f"Unknown document IDs: {', '.join(unknown)}")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pdf_dir / "download_manifest.jsonl"
    known_manifest_ids = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                known_manifest_ids.add(json.loads(line).get("doc_id"))
            except json.JSONDecodeError:
                continue
    for index, doc_id in enumerate(doc_ids):
        target = pdf_dir / f"{doc_id}.pdf"
        if target.is_file() and target.stat().st_size:
            body = target.read_bytes()
            if not body.startswith(b"%PDF-"):
                raise RuntimeError(f"Cached file for {doc_id} is not a PDF: {target}")
            if doc_id not in known_manifest_ids:
                manifest = {
                    "doc_id": doc_id,
                    "url": metadata[doc_id]["url"].strip(),
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "recording_method": "preexisting_cache_observed",
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            print(f"cached {doc_id}: {target.stat().st_size} bytes")
            continue
        if index:
            time.sleep(max(0.0, delay))
        url = metadata[doc_id]["url"].strip()
        request = urllib.request.Request(url, headers={"User-Agent": "wattbot-rag/0.1 (polite single-document fetch)"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Download failed for {doc_id} ({url}): {exc}") from exc
        if not body.startswith(b"%PDF-"):
            raise RuntimeError(
                f"{doc_id} did not return a PDF (Content-Type={content_type!r}); not cached."
            )
        temp_target = target.with_suffix(".pdf.tmp")
        temp_target.write_bytes(body)
        temp_target.replace(target)
        manifest = {
            "doc_id": doc_id,
            "url": url,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_method": "downloaded",
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        print(f"downloaded {doc_id}: {len(body)} bytes from pinned URL")

