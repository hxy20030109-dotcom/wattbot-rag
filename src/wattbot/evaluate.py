"""Deterministic held-out retrieval evaluation against training citations."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from wattbot.data import flag, parse_refs, read_csv
from wattbot.retrieval import BM25


def is_holdout(question_id: str, modulus: int = 5) -> bool:
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulus == 0


def evaluate(data_dir: Path, chunks_path: Path, ks: list[int]) -> dict:
    if not ks or any(k < 1 for k in ks):
        raise ValueError("All k values must be positive")
    questions = read_csv(data_dir / "train_QA.csv")
    holdout = [row for row in questions if is_holdout(row["id"]) ]
    index_started = time.perf_counter()
    model = BM25.from_jsonl(chunks_path)
    index_seconds = time.perf_counter() - index_started
    query_ms = []
    available_docs = {str(doc.get("doc_id", "")).casefold() for doc in model.documents}
    by_type: dict[str, list[tuple[set[str], list[dict]]]] = defaultdict(list)
    eligible_by_type: dict[str, list[tuple[set[str], list[dict]]]] = defaultdict(list)
    gold_ref_total = 0
    gold_ref_available = 0
    fully_covered = 0
    for row in holdout:
        gold = set(parse_refs(row.get("ref_id")))
        if not gold:
            continue
        gold_ref_total += len(gold)
        gold_ref_available += len(gold & available_docs)
        covered = gold <= available_docs
        fully_covered += covered
        query_started = time.perf_counter()
        predicted = model.search_documents(row["question"], max(ks))
        query_ms.append((time.perf_counter() - query_started) * 1000)
        by_type["ALL"].append((gold, predicted))
        if covered:
            eligible_by_type["ALL"].append((gold, predicted))
        for label in ("Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile"):
            if flag(row.get(label)):
                by_type[label].append((gold, predicted))
                if covered:
                    eligible_by_type[label].append((gold, predicted))

    def summarize(groups: dict[str, list[tuple[set[str], list[dict]]]]) -> dict:
        metrics = {}
        for label, examples in groups.items():
            label_scores = {}
            for k in ks:
                recalls, reciprocal_ranks = [], []
                for gold, predicted in examples:
                    top = [str(item["doc_id"]).casefold() for item in predicted[:k]]
                    recalls.append(len(gold & set(top)) / len(gold))
                    rank = next((i + 1 for i, doc_id in enumerate(top) if doc_id in gold), None)
                    reciprocal_ranks.append(1 / rank if rank else 0.0)
                label_scores[str(k)] = {
                    "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
                    "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
                }
            metrics[label] = {"n": len(examples), **label_scores}
        return metrics

    return {
        "method": "bm25",
        "parameters": {"k1": model.k1, "b": model.b, "document_ranking": "best_scoring_chunk"},
        "snapshot_sha256": {
            "metadata_csv": hashlib.sha256((data_dir / "metadata.csv").read_bytes()).hexdigest(),
            "train_qa_csv": hashlib.sha256((data_dir / "train_QA.csv").read_bytes()).hexdigest(),
            "chunks_jsonl": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        },
        "split": "first_4_sha256_bytes_of_question_id_as_big_endian_integer_mod_5_equals_0",
        "train_rows": len(questions),
        "holdout_rows": len(holdout),
        "answerable_holdout_rows": len(by_type["ALL"]),
        "indexed_documents": len(available_docs),
        "gold_reference_coverage": gold_ref_available / gold_ref_total if gold_ref_total else 0.0,
        "fully_covered_questions": fully_covered,
        "latency": {
            "index_build_seconds": index_seconds,
            "measured_queries": len(query_ms),
            "mean_query_ms": statistics.mean(query_ms) if query_ms else None,
            "median_query_ms": statistics.median(query_ms) if query_ms else None,
            "p95_query_ms": sorted(query_ms)[math.ceil(0.95 * len(query_ms)) - 1] if query_ms else None,
            "scope": "BM25 document retrieval only; excludes LLM generation and PDF parsing",
        },
        "retrieval_metrics_all_holdout": summarize(by_type),
        "retrieval_metrics_fully_covered_only": summarize(eligible_by_type),
    }

