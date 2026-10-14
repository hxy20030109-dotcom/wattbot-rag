"""Audit the downloaded Kaggle files without changing them."""

from __future__ import annotations

import hashlib
from pathlib import Path

from wattbot.data import fieldnames, flag, is_blank, read_csv


SNAPSHOT = {
    "metadata.csv": (122, {"id", "url", "title"}),
    "train_QA.csv": (245, {"id", "question", "answer_value", "answer_unit", "ref_id", "is_NA"}),
    "test_Q.csv": (317, {"id", "question", "answer_unit"}),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(data_dir: Path) -> int:
    problems: list[str] = []
    print(f"Data directory: {data_dir.resolve()}")
    for name, (expected_rows, required_columns) in SNAPSHOT.items():
        path = data_dir / name
        if not path.is_file():
            print(f"MISSING {name}")
            problems.append(name)
            continue
        rows = read_csv(path)
        columns = fieldnames(path)
        print(
            f"{name}: rows={len(rows)}, columns={len(columns)}, "
            f"sha256={sha256(path)}"
        )
        missing_columns = required_columns - set(columns)
        if missing_columns:
            problems.append(f"missing columns for {name}: {sorted(missing_columns)}")
        if len(rows) != expected_rows:
            print(
                f"  NOTE: row count differs from the initial ZIP snapshot ({expected_rows}); "
                "Kaggle says questions may be added or corrected during the competition."
            )

    train_path = data_dir / "train_QA.csv"
    metadata_path = data_dir / "metadata.csv"
    if train_path.is_file():
        train = read_csv(train_path)
        na_rows = [row for row in train if flag(row.get("is_NA"))]
        blank_rows = [row for row in train if is_blank(row.get("answer_value"))]
        print(f"Training abstentions: is_NA={len(na_rows)}, blank answer_value={len(blank_rows)}")
        if len(na_rows) != len(blank_rows):
            problems.append("is_NA and blank answer_value counts differ")
        if len(na_rows) != 25:
            print(
                "  NOTE: Kaggle Data page currently says 25 intentional unanswerable questions; "
                f"this CSV snapshot has {len(na_rows)} flagged rows."
            )

    if metadata_path.is_file():
        metadata = read_csv(metadata_path)
        ids = [row.get("id", "") for row in metadata]
        urls = [row.get("url", "") for row in metadata]
        print(
            f"Metadata: unique_ids={len(set(ids))}, blank_urls={sum(not u.strip() for u in urls)}"
        )
        if len(set(ids)) != len(metadata) or any(not url.strip() for url in urls):
            problems.append("metadata IDs or URLs are invalid")

    return 1 if problems else 0

