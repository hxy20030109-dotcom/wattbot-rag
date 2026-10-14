"""Rank-fusion retrieval baseline combining BM25 with dense cosine search."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from wattbot.data import flag, parse_refs, read_csv
from wattbot.dense_retrieval import DenseIndex
from wattbot.evaluate import is_holdout
from wattbot.retrieval import BM25


def reciprocal_rank_fusion(rankings: list[list[dict]], k0: int = 60) -> list[dict]:
    """Fuse ranked documents without comparing BM25 and cosine score scales."""
    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, dict] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            doc_id = str(item.get("doc_id", "")).casefold()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k0 + rank)
            representatives.setdefault(doc_id, dict(item))
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    output = []
    for doc_id in ranked:
        item = representatives[doc_id]
        item["score"] = scores[doc_id]
        output.append(item)
    return output


class DenseChunkRetriever:
    """Generation-time dense chunk retriever with lexical IDF for sentence selection."""

    def __init__(self, dense: DenseIndex, chunks_path: Path):
        self.dense = dense
        self.lexical = BM25.from_jsonl(chunks_path)
        self.idf = self.lexical.idf

    def search(self, query: str, k: int = 8) -> list[dict]:
        chunks, _ = self.dense.score_chunks(query)
        return chunks[:k]

    def close(self) -> None:
        self.dense.close()


class HybridChunkRetriever(DenseChunkRetriever):
    """RRF-fuse dense and BM25 chunk rankings for evidence passage selection."""

    def __init__(self, dense: DenseIndex, chunks_path: Path, rrf_k: int = 60):
        super().__init__(dense, chunks_path)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 8) -> list[dict]:
        dense_chunks, _ = self.dense.score_chunks(query)
        sparse_chunks = self.lexical.search(query, len(self.lexical.documents))
        fused: dict[str, float] = defaultdict(float)
        representatives: dict[str, dict] = {}
        for ranked in (dense_chunks, sparse_chunks):
            for rank, chunk in enumerate(ranked, start=1):
                chunk_id = str(chunk.get("chunk_id", ""))
                if not chunk_id:
                    continue
                fused[chunk_id] += 1.0 / (self.rrf_k + rank)
                representatives.setdefault(chunk_id, dict(chunk))
        result = []
        for chunk_id in sorted(fused, key=lambda item: (-fused[item], item))[:k]:
            chunk = representatives[chunk_id]
            chunk["score"] = fused[chunk_id]
            result.append(chunk)
        return result


def evaluate_hybrid(data_dir: Path, chunks_path: Path, vectors_path: Path, ks: list[int], rrf_k: int = 60) -> dict:
    """Measure BM25+dense RRF on the same fixed training holdout."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("All k values must be positive")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    questions = read_csv(data_dir / "train_QA.csv")
    holdout = [row for row in questions if is_holdout(row["id"])]
    load_started = time.perf_counter()
    dense = DenseIndex(vectors_path, chunks_path)
    bm25 = BM25.from_jsonl(chunks_path)
    load_seconds = time.perf_counter() - load_started
    query_ms: list[float] = []
    max_docs = len({str(item.get("doc_id", "")).casefold() for item in dense.chunks})
    groups: dict[str, list[tuple[set[str], list[dict]]]] = defaultdict(list)
    covered_groups: dict[str, list[tuple[set[str], list[dict]]]] = defaultdict(list)
    available_docs = {str(item.get("doc_id", "")).casefold() for item in dense.chunks}
    gold_total = gold_available = fully_covered = 0
    try:
        for row in holdout:
            gold = set(parse_refs(row.get("ref_id")))
            if not gold:
                continue
            gold_total += len(gold)
            gold_available += len(gold & available_docs)
            covered = gold <= available_docs
            fully_covered += int(covered)
            started = time.perf_counter()
            dense_docs, _ = dense.search_documents(row["question"], max_docs)
            sparse_docs = bm25.search_documents(row["question"], max_docs)
            fused = reciprocal_rank_fusion([dense_docs, sparse_docs], rrf_k)
            query_ms.append((time.perf_counter() - started) * 1000)
            groups["ALL"].append((gold, fused))
            if covered:
                covered_groups["ALL"].append((gold, fused))
            for label in ("Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile"):
                if flag(row.get(label)):
                    groups[label].append((gold, fused))
                    if covered:
                        covered_groups[label].append((gold, fused))
    finally:
        dense.close()

    def summarize(by_type: dict[str, list[tuple[set[str], list[dict]]]]) -> dict:
        result = {}
        for label, examples in by_type.items():
            scores = {}
            for k in ks:
                recalls, reciprocal_ranks = [], []
                for gold, predicted in examples:
                    ids = [str(item["doc_id"]).casefold() for item in predicted[:k]]
                    recalls.append(len(gold & set(ids)) / len(gold))
                    rank = next((i + 1 for i, doc_id in enumerate(ids) if doc_id in gold), None)
                    reciprocal_ranks.append(1 / rank if rank else 0.0)
                scores[str(k)] = {
                    "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
                    "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
                }
            result[label] = {"n": len(examples), **scores}
        return result

    index_meta = json.loads(vectors_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return {
        "method": "hybrid_reciprocal_rank_fusion",
        "parameters": {"fusion": "sum(1/(rrf_k + rank))", "rrf_k": rrf_k,
                        "dense_model": index_meta["model"], "dense_model_digest": index_meta["model_digest"],
                        "dense_dimensions": index_meta["dimensions"], "candidate_documents": max_docs,
                        "bm25": {"k1": bm25.k1, "b": bm25.b}},
        "snapshot_sha256": {
            "metadata_csv": hashlib.sha256((data_dir / "metadata.csv").read_bytes()).hexdigest(),
            "train_qa_csv": hashlib.sha256((data_dir / "train_QA.csv").read_bytes()).hexdigest(),
            "chunks_jsonl": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
            "embedding_index_sha256": index_meta["index_sha256"],
        },
        "split": "first_4_sha256_bytes_of_question_id_as_big_endian_integer_mod_5_equals_0",
        "train_rows": len(questions), "holdout_rows": len(holdout),
        "answerable_holdout_rows": len(groups["ALL"]), "indexed_documents": max_docs,
        "gold_reference_coverage": gold_available / gold_total if gold_total else 0.0,
        "fully_covered_questions": fully_covered,
        "latency": {"index_load_seconds": load_seconds, "measured_queries": len(query_ms),
                    "mean_query_ms": statistics.mean(query_ms) if query_ms else None,
                    "median_query_ms": statistics.median(query_ms) if query_ms else None,
                    "p95_query_ms": sorted(query_ms)[math.ceil(.95 * len(query_ms)) - 1] if query_ms else None,
                    "scope": "local query embedding, BM25, dense ranking, and RRF; excludes answer generation"},
        "retrieval_metrics_all_holdout": summarize(groups),
        "retrieval_metrics_fully_covered_only": summarize(covered_groups),
    }
