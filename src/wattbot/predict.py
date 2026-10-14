"""Generate an official-format CSV using retrieved evidence and an optional LLM."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wattbot.data import read_csv
from wattbot.generation import SYSTEM_PROMPT, answer_question
from wattbot.retrieval import BM25


OUTPUT_FIELDS = [
    "id",
    "question",
    "answer",
    "answer_value",
    "answer_unit",
    "ref_id",
    "ref_url",
    "supporting_materials",
    "explanation",
]


def predict(input_path: Path, chunks_path: Path, metadata_path: Path, output_path: Path, top_k: int = 8, *, resume: bool = False) -> dict:
    wall_started = time.perf_counter()
    if top_k < 1:
        raise ValueError("top_k must be positive")
    questions = read_csv(input_path)
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    metadata = {row["id"]: row for row in read_csv(metadata_path)}
    index = BM25.from_jsonl(chunks_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_suffix(".trace.jsonl")
    run_path = output_path.with_suffix(".run.json")
    base = os.environ.get("WATTBOT_LLM_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("WATTBOT_LLM_MODEL", "").strip()
    if not base or not model:
        raise ValueError("Configure the local/API model before starting predictions")
    local = urllib.parse.urlsplit(base).hostname in {"localhost", "127.0.0.1", "::1"}
    runtime = {}
    if local and base.endswith("/v1"):
        with urllib.request.urlopen(base[:-3] + "/api/tags", timeout=15) as response:
            tags = json.load(response)
        selected = next((item for item in tags["models"] if item["name"] == model), None)
        if selected is None:
            raise ValueError(f"Local model not installed: {model}")
        with urllib.request.urlopen(base[:-3] + "/api/version", timeout=15) as response:
            runtime["ollama_version"] = json.load(response)["version"]
        runtime.update({"model_digest": selected["digest"], "model_details": selected.get("details")})
    settings = {
        "model": model, "base_url": base, "top_k": top_k, "temperature": 0, "seed": 42,
        "max_tokens": int(os.environ.get("WATTBOT_LLM_MAX_TOKENS", "4096")),
        "backend": os.environ.get("WATTBOT_LLM_BACKEND", "openai_compatible"),
        "context_length": int(os.environ.get("WATTBOT_LLM_CONTEXT", "8192")),
        "thinking": os.environ.get("WATTBOT_LLM_THINK", "low"),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "files_sha256": {"questions": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                         "metadata": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                         "chunks": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
                         "generation_py": hashlib.sha256(Path(__file__).with_name("generation.py").read_bytes()).hexdigest(),
                         "retrieval_py": hashlib.sha256(Path(__file__).with_name("retrieval.py").read_bytes()).hexdigest()},
        "runtime": runtime,
    }
    completed = set()
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"Prediction file exists; use --resume to continue: {output_path}")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError("Cannot resume with different data, model, prompt, code or parameters")
        completed = {row["id"] for row in read_csv(output_path)}
        trace_ids = {json.loads(line)["id"] for line in trace_path.read_text(encoding="utf-8").splitlines() if line}
        if completed != trace_ids:
            raise ValueError("CSV and trace checkpoints differ; inspect before resuming")
    else:
        if trace_path.exists() or run_path.exists():
            raise FileExistsError("Orphaned run metadata exists; use a new output filename")
        run_path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "settings": settings}, indent=2), encoding="utf-8")
    mode = "a" if completed else "w"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as handle, trace_path.open(mode, encoding="utf-8") as trace_file:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if not completed:
            writer.writeheader()
            handle.flush()
        for position, row in enumerate(questions, start=1):
            if row["id"] in completed:
                continue
            trace = {"id": row["id"], "question": row["question"], "expected_answer_unit": row.get("answer_unit", "")}
            try:
                result, elapsed = answer_question(row["question"], index, metadata, top_k,
                                                  answer_unit=row.get("answer_unit", ""), trace=trace)
            except Exception as exc:
                error = {"id": row["id"], "error": f"{type(exc).__name__}: {exc}",
                         "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
                with output_path.with_suffix(".errors.jsonl").open("a", encoding="utf-8") as error_file:
                    error_file.write(json.dumps(error) + "\n")
                raise
            result.update({"id": row["id"], "question": row["question"]})
            if row.get("answer_unit"):
                result["answer_unit"] = row["answer_unit"]
            writer.writerow(result)
            handle.flush()
            trace["prediction"] = result
            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_file.flush()
            print(f"[{position}/{len(questions)}] {row['id']} {trace['decision']} {elapsed:.2f}s", flush=True)
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    timings = [row["elapsed_seconds"] for row in traces]
    summary = {
        "questions": len(questions),
        "abstentions": sum(1 for row in read_csv(output_path) if row["answer_value"] == "is_blank"),
        "total_seconds": sum(timings),
        "mean_seconds": sum(timings) / len(timings) if timings else 0,
        "median_seconds": statistics.median(timings) if timings else 0,
        "p95_seconds": sorted(timings)[math.ceil(.95 * len(timings)) - 1] if timings else 0,
        "wall_seconds_this_invocation": time.perf_counter() - wall_started,
        "decisions": dict(Counter(row["decision"] for row in traces)),
        "token_usage_reported_by_endpoint": {
            key: sum((row.get("llm_response", {}).get("usage") or {}).get(key, 0) for row in traces)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "external_api_cost_usd": 0 if local else None,
        "cost_scope": "Local inference: no paid API calls; electricity and hardware costs not measured" if local else "Provider pricing not configured",
        "output": str(output_path.resolve()),
    }
    output_path.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def holdout_predictions(data_dir: Path, chunks_path: Path, output_dir: Path, top_k: int = 8, *, resume: bool = False) -> tuple[Path, Path]:
    from wattbot.evaluate import is_holdout

    all_rows = read_csv(data_dir / "train_QA.csv")
    holdout_rows = [row for row in all_rows if is_holdout(row["id"])]
    solution_path = output_dir / "holdout_solution.csv"
    question_path = output_dir / "holdout_questions.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with solution_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(holdout_rows[0]))
        writer.writeheader()
        writer.writerows(holdout_rows)
    with question_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "answer_unit"])
        writer.writeheader()
        for row in holdout_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    prediction_path = output_dir / "holdout_predictions.csv"
    predict(question_path, chunks_path, data_dir / "metadata.csv", prediction_path, top_k, resume=resume)
    return solution_path, prediction_path

