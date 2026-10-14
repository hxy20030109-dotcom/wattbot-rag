"""Run the unmodified shipped scorer and persist per-question diagnostics."""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--solution", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
args = parser.parse_args()
scorer_path = root / "data/competition/Score.py"
spec = importlib.util.spec_from_file_location("official_wattbot_score", scorer_path)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
solution = pd.read_csv(args.solution)
predictions = pd.read_csv(args.predictions)
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    result = scorer.score(solution.copy(), predictions.copy(), "id")
breakdown = copy.deepcopy(scorer.LAST_BREAKDOWN)
text = buffer.getvalue()
print(text)
args.predictions.with_suffix(".score.txt").write_text(text, encoding="utf-8")
summary = {"official_score": result, "official_breakdown": breakdown,
           "scorer_sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
           "solution_sha256": hashlib.sha256(args.solution.read_bytes()).hexdigest(),
           "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
args.predictions.with_suffix(".score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
trace_path = args.predictions.with_suffix(".trace.jsonl")
traces = {row["id"]: row for row in (json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines())} if trace_path.exists() else {}
diagnostics = []
for _, gold in solution.iterrows():
    pred = predictions[predictions["id"] == gold["id"]]
    row_score = scorer.score(pd.DataFrame([gold]), pred.copy(), "id", verbose=False)
    row_breakdown = scorer.LAST_BREAKDOWN
    trace = traces.get(str(gold["id"]), {})
    retrieved = {item["doc_id"] for item in trace.get("retrieved_passages", [])}
    gold_refs = set() if scorer.is_blank(gold["ref_id"]) else set(scorer.canon_refs(gold["ref_id"]))
    diagnostics.append({"id": gold["id"], "question": gold["question"],
                        "gold_value": str(gold["answer_value"]), "predicted_value": str(pred.iloc[0]["answer_value"]),
                        "value_correct": row_breakdown["per_type"]["ALL"]["wrong"] == 0,
                        "gold_ref_ids": sorted(gold_refs), "predicted_ref_ids": [] if scorer.is_blank(pred.iloc[0]["ref_id"]) else scorer.canon_refs(pred.iloc[0]["ref_id"]),
                        "retrieved_gold_ref_fraction": len(gold_refs & retrieved) / len(gold_refs) if gold_refs else None,
                        "decision": trace.get("decision"), "official_row_score": row_score,
                        "ref_f1": row_breakdown["ref_f1"], "gold_unanswerable": scorer.is_blank(gold["answer_value"])})
args.predictions.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"official_score": result, "value_correct": sum(row["value_correct"] for row in diagnostics),
                  "questions": len(diagnostics)}, indent=2))
