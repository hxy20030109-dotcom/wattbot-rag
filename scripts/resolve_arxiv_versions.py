"""Record the highest version present in sanctioned GCS for versionless URLs.

This does not claim to recover the version the competition authors originally read.
The original URLs have no version; that ambiguity remains visible in the report.
"""

import csv
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with (root / "data/competition/metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
selected = {}
for row in rows:
    match = re.fullmatch(r"https://arxiv.org/pdf/(\d{4}\.\d{4,5})", row["url"].strip())
    if match:
        selected[match.group(1)] = row["id"]
versions = {}
listings = []
for identifier, doc_id in selected.items():
    prefix = f"arxiv/arxiv/pdf/{identifier[:4]}/{identifier}v"
    query = urllib.parse.urlencode({"prefix": prefix, "maxResults": 1000,
                                   "fields": "items(name,generation,size),nextPageToken"})
    url = "https://storage.googleapis.com/storage/v1/b/arxiv-dataset/o?" + query
    with urllib.request.urlopen(url, timeout=45) as response:
        listing = json.load(response)
    if listing.get("nextPageToken"):
        raise ValueError(f"Too many versions returned for {doc_id}")
    candidates = []
    for item in listing.get("items", []):
        match = re.fullmatch(re.escape(prefix) + r"(\d+)\.pdf", item["name"])
        if match:
            candidates.append(int(match.group(1)))
    if candidates:
        versions[doc_id] = f"{identifier}v{max(candidates)}"
    listings.append({"doc_id": doc_id, "listing_url": url, "available_versions": sorted(candidates)})
    print(f"{doc_id}: {versions.get(doc_id, 'unavailable')}", flush=True)
report = {"resolved_at_utc": datetime.now(timezone.utc).isoformat(),
          "strategy": "highest_version_available_in_sanctioned_mirror",
          "limitation": "Original metadata URLs omit versions; this does not verify the competition authors' snapshot.",
          "versions": versions, "unresolved": sorted(set(selected.values()) - versions.keys()), "listings": listings}
out = root / "data/pdfs/unpinned_version_resolution.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
