"""Select a deterministic, small development-only plumbing set."""

import csv
import argparse
import hashlib
import json
from pathlib import Path

from wattbot.data import flag, is_blank, read_csv
from wattbot.evaluate import is_holdout

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=root / "outputs/development_v1")
parser.add_argument("--all", action="store_true", help="include every non-holdout training row")
args = parser.parse_args()
rows = [row for row in read_csv(root / "data/competition/train_QA.csv") if not is_holdout(row["id"])]
ordered = sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
if args.all:
    selected = ordered
else:
    selected = []
    for label in ("Quote", "Table", "Figure", "Math", "CrossPaper", "Reconcile"):
        candidate = next((row for row in ordered if flag(row.get(label)) and not is_blank(row["answer_value"])
                          and row["id"] not in {item["id"] for item in selected}), None)
        if candidate:
            selected.append(candidate)
    for row in ordered:
        if len(selected) >= 8:
            break
        if not is_blank(row["answer_value"]) and row["id"] not in {item["id"] for item in selected}:
            selected.append(row)
    selected.extend(row for row in ordered if is_blank(row["answer_value"]))
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
with (out / "solution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
with (out / "questions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["id", "question", "answer_unit"])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in selected)
print(json.dumps({"development_population": len(rows), "selected_ids": [row["id"] for row in selected],
                  "selected_rows": len(selected), "holdout_rows_included": sum(is_holdout(row["id"]) for row in selected)}, indent=2))
