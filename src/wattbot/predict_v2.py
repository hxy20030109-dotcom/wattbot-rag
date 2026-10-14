"""Run the sentence-ID evidence generation pipeline into a fresh CSV and traces."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wattbot.data import read_csv
from wattbot.generation import SYSTEM_PROMPT
from wattbot.generation_v2 import SYSTEM_PROMPT_V2, answer_question_v2
from wattbot.dense_retrieval import DenseIndex
from wattbot.hybrid_retrieval import DenseChunkRetriever, HybridChunkRetriever
from wattbot.predict import OUTPUT_FIELDS
from wattbot.retrieval import BM25


def predict_v2(input_path: Path, chunks_path: Path, metadata_path: Path, output_path: Path,
               top_k: int = 8, *, resume: bool = False, retrieval: str = "bm25",
               vectors_path: Path = Path("data/indexes/nomic-embed-text-v1.5.npy"),
               rrf_k: int = 60) -> dict:
    started = time.perf_counter()
    if top_k < 1:
        raise ValueError("top_k must be positive")
    questions = read_csv(input_path)
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    metadata = {row["id"]: row for row in read_csv(metadata_path)}
    if retrieval not in {"bm25", "dense", "hybrid"}:
        raise ValueError("retrieval must be one of: bm25, dense, hybrid")
    if retrieval == "bm25":
        index = BM25.from_jsonl(chunks_path)
    else:
        dense_index = DenseIndex(vectors_path, chunks_path)
        if retrieval == "dense":
            index = DenseChunkRetriever(dense_index, chunks_path)
        else:
            index = HybridChunkRetriever(dense_index, chunks_path, rrf_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_suffix(".trace.jsonl")
    run_path = output_path.with_suffix(".run.json")
    metrics_path = output_path.with_suffix(".metrics.json")
    settings = {
        "experiment": "sentence_id_evidence_v2", "retrieval": retrieval, "top_k": top_k,
        "max_evidence_sentences_per_passage": 6,
        "model": os.environ.get("WATTBOT_LLM_MODEL", ""),
        "base_url": os.environ.get("WATTBOT_LLM_BASE_URL", ""),
        "backend": os.environ.get("WATTBOT_LLM_BACKEND", "openai_compatible"),
        "context_length": int(os.environ.get("WATTBOT_LLM_CONTEXT", "8192")),
        "max_tokens": int(os.environ.get("WATTBOT_LLM_MAX_TOKENS", "2048")),
        "thinking": os.environ.get("WATTBOT_LLM_THINK", "low"),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT_V2.encode()).hexdigest(),
        "baseline_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "generation_v2_sha256": hashlib.sha256(Path(__file__).with_name("generation_v2.py").read_bytes()).hexdigest(),
        "retrieval_py_sha256": hashlib.sha256(Path(__file__).with_name("retrieval.py").read_bytes()).hexdigest(),
    }
    if retrieval != "bm25":
        settings["embedding_index_sha256"] = json.loads(
            vectors_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )["index_sha256"]
        settings["dense_retrieval_py_sha256"] = hashlib.sha256(
            Path(__file__).with_name("dense_retrieval.py").read_bytes()
        ).hexdigest()
    if retrieval == "hybrid":
        settings["rrf_k"] = rrf_k
        settings["hybrid_retrieval_py_sha256"] = hashlib.sha256(
            Path(__file__).with_name("hybrid_retrieval.py").read_bytes()
        ).hexdigest()
    base_url = settings["base_url"].rstrip("/")
    local = base_url.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]"))
    runtime = {}
    if local and base_url.endswith("/v1"):
        api_root = base_url[:-3]
        with urllib.request.urlopen(api_root + "/api/tags", timeout=15) as response:
            tags = json.load(response)
        selected = next((item for item in tags.get("models", []) if item.get("name") == settings["model"]), None)
        if selected is None:
            raise ValueError(f"Local model is not installed: {settings['model']}")
        with urllib.request.urlopen(api_root + "/api/version", timeout=15) as response:
            runtime["ollama_version"] = json.load(response).get("version")
        runtime.update({"model_digest": selected.get("digest"), "model_details": selected.get("details")})
    settings["runtime"] = runtime
    completed = set()
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"Output exists; use --resume or a new path: {output_path}")
        if not run_path.exists() or not trace_path.exists():
            raise ValueError("Cannot resume: V2 run metadata or trace is missing")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError("Cannot resume with different data, model, prompt, code or parameters")
        completed = {row["id"] for row in read_csv(output_path)}
        trace_ids = {json.loads(line)["id"] for line in trace_path.read_text(encoding="utf-8").splitlines() if line}
        if completed != trace_ids or not completed.issubset({row["id"] for row in questions}):
            raise ValueError("CSV and trace checkpoints differ or include unexpected IDs")
        if len(completed) == len(questions):
            raise FileExistsError("V2 run is already complete; choose a new output path for another run")
    else:
        if trace_path.exists() or run_path.exists() or metrics_path.exists():
            raise FileExistsError("Orphaned V2 sidecar exists; use a new output filename")
        run_path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "settings": settings}, indent=2), encoding="utf-8")
    invocation_started = time.perf_counter()
    mode = "a" if completed else "w"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as output, \
            trace_path.open(mode, encoding="utf-8") as trace_file:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        if not completed:
            writer.writeheader()
            output.flush()
        for position, row in enumerate(questions, start=1):
            if row["id"] in completed:
                continue
            trace = {"id": row["id"], "question": row["question"],
                     "expected_answer_unit": row.get("answer_unit", "")}
            try:
                result, elapsed = answer_question_v2(row["question"], index, metadata, top_k,
                                                      answer_unit=row.get("answer_unit", ""), trace=trace)
            except Exception as exc:
                error_path = output_path.with_suffix(".errors.jsonl")
                with error_path.open("a", encoding="utf-8") as error_file:
                    error_file.write(json.dumps({"id": row["id"],
                                                 "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                raise
            result.update({"id": row["id"], "question": row["question"]})
            if row.get("answer_unit"):
                result["answer_unit"] = row["answer_unit"]
            writer.writerow(result)
            output.flush()
            trace["prediction"] = result
            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_file.flush()
            print(f"[{position}/{len(questions)}] {row['id']} {trace['decision']} {elapsed:.2f}s", flush=True)

    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    timings = [row["elapsed_seconds"] for row in traces]
    decisions = Counter(row["decision"] for row in traces)
    token_usage = Counter()
    for row in traces:
        usage = row.get("llm_response", {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            token_usage[key] += usage.get(key, 0) or 0
    sorted_times = sorted(timings)
    metrics = {
        "questions": len(questions), "abstentions": sum(decisions[key] for key in decisions
                                                            if key != "accepted"),
        "total_seconds": sum(timings), "mean_seconds": statistics.mean(timings) if timings else 0,
        "median_seconds": statistics.median(timings) if timings else 0,
        "p95_seconds": sorted_times[math.ceil(.95 * len(sorted_times)) - 1] if timings else 0,
        "wall_seconds": time.perf_counter() - started,
        "wall_seconds_this_invocation": time.perf_counter() - invocation_started,
        "decisions": dict(decisions),
        "token_usage_reported_by_endpoint": dict(token_usage),
        "external_api_cost_usd": 0 if local else None,
        "output": str(output_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    close = getattr(index, "close", None)
    if close is not None:
        close()
    return metrics
