"""Dense cosine retrieval over a frozen Ollama embedding index."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np

from wattbot.embeddings import DEFAULT_MODEL, OllamaEmbedder


class DenseIndex:
    def __init__(self, vectors_path: Path, chunks_path: Path):
        self.metadata = json.loads(vectors_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        self.vectors = np.load(vectors_path, mmap_mode="r")
        with chunks_path.open(encoding="utf-8") as handle:
            self.chunks = [json.loads(line) for line in handle if line.strip()]
        if len(self.chunks) != len(self.vectors) or len(self.chunks) != self.metadata["chunk_count"]:
            raise ValueError("Embedding vectors and chunk snapshot have different row counts")
        actual_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        if actual_hash != self.metadata["chunks_sha256"]:
            raise ValueError("Chunk file hash does not match the frozen embedding index")
        recorded_ids = self.metadata.get("chunk_ids")
        if recorded_ids is not None and recorded_ids != [str(item.get("chunk_id", "")) for item in self.chunks]:
            raise ValueError("Chunk order does not match the frozen embedding index")
        self.embedder = OllamaEmbedder(self.metadata["model"])

    def close(self) -> None:
        self.embedder.close()
        mapping = getattr(self.vectors, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def score_chunks(self, query: str) -> tuple[list[dict], dict]:
        started = time.perf_counter()
        vector = self.embedder.embed(query, "search_query")
        embedded_at = time.perf_counter()
        if vector.shape != (self.vectors.shape[1],):
            raise ValueError("Query vector dimensions do not match the corpus index")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Query embedding is a zero vector")
        scores = np.asarray(self.vectors @ (vector / norm), dtype=np.float32)
        ranked = np.argsort(-scores, kind="stable")
        results = []
        for index in ranked:
            item = dict(self.chunks[int(index)])
            item["score"] = float(scores[index])
            results.append(item)
        finished = time.perf_counter()
        return results, {"embed_ms": (embedded_at - started) * 1000,
                         "rank_ms": (finished - embedded_at) * 1000,
                         "total_ms": (finished - started) * 1000}

    def search_documents(self, query: str, k: int = 10) -> tuple[list[dict], dict]:
        chunks, timing = self.score_chunks(query)
        best = {}
        for item in chunks:
            doc_id = str(item.get("doc_id", "")).casefold()
            if doc_id and doc_id not in best:
                best[doc_id] = item
                if len(best) == k:
                    break
        return list(best.values()), timing


def evaluate_dense(data_dir: Path, chunks_path: Path, vectors_path: Path, ks: list[int]) -> dict:
    """Compare dense document retrieval against gold source IDs on the fixed holdout."""
    from collections import defaultdict
    import hashlib
    import math
    import statistics

    from wattbot.data import flag, parse_refs, read_csv
    from wattbot.evaluate import is_holdout

    if not ks or any(k < 1 for k in ks):
        raise ValueError("All k values must be positive")
    holdout = [row for row in read_csv(data_dir / "train_QA.csv") if is_holdout(row["id"])]
    index_started = time.perf_counter()
    model = DenseIndex(vectors_path, chunks_path)
    index_seconds = time.perf_counter() - index_started
    query_ms, embedding_ms, ranking_ms = [], [], []
    available_docs = {str(row.get("doc_id", "")).casefold() for row in model.chunks}
    by_type: dict[str, list[tuple[set[str], list[dict]]]] = defaultdict(list)
    eligible: dict[str, list[tuple[set[str], list[dict]]]] = defaultdict(list)
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
            predicted, timing = model.search_documents(row["question"], max(ks))
            query_ms.append(timing["total_ms"])
            embedding_ms.append(timing["embed_ms"])
            ranking_ms.append(timing["rank_ms"])
            by_type["ALL"].append((gold, predicted))
            if covered:
                eligible["ALL"].append((gold, predicted))
            for label in ("Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile"):
                if flag(row.get(label)):
                    by_type[label].append((gold, predicted))
                    if covered:
                        eligible[label].append((gold, predicted))
    finally:
        model.close()

    def summarize(groups: dict[str, list[tuple[set[str], list[dict]]]]) -> dict:
        summary = {}
        for label, examples in groups.items():
            scores = {}
            for k in ks:
                recalls, reciprocal = [], []
                for gold, predicted in examples:
                    ids = [str(item["doc_id"]).casefold() for item in predicted[:k]]
                    recalls.append(len(gold & set(ids)) / len(gold))
                    rank = next((i + 1 for i, doc_id in enumerate(ids) if doc_id in gold), None)
                    reciprocal.append(1 / rank if rank else 0.0)
                scores[str(k)] = {"recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
                                 "mrr_at_k": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0}
            summary[label] = {"n": len(examples), **scores}
        return summary

    vector_meta = json.loads(vectors_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    snapshot = {"metadata_csv": hashlib.sha256((data_dir / "metadata.csv").read_bytes()).hexdigest(),
                "train_qa_csv": hashlib.sha256((data_dir / "train_QA.csv").read_bytes()).hexdigest(),
                "chunks_jsonl": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
                "embedding_index_sha256": vector_meta["index_sha256"]}
    return {
        "method": "dense_cosine",
        "parameters": {"model": vector_meta["model"], "model_digest": vector_meta["model_digest"],
                        "dimensions": vector_meta["dimensions"], "chunk_ranking": "cosine_then_max_per_document",
                        "query_prefix": "search_query: ", "document_prefix": "search_document: "},
        "snapshot_sha256": snapshot,
        "split": "first_4_sha256_bytes_of_question_id_as_big_endian_integer_mod_5_equals_0",
        "train_rows": len(read_csv(data_dir / "train_QA.csv")), "holdout_rows": len(holdout),
        "answerable_holdout_rows": len(by_type["ALL"]), "indexed_documents": len(available_docs),
        "gold_reference_coverage": gold_available / gold_total if gold_total else 0,
        "fully_covered_questions": fully_covered,
        "latency": {"index_load_seconds": index_seconds, "measured_queries": len(query_ms),
                    "mean_query_ms": statistics.mean(query_ms) if query_ms else None,
                    "median_query_ms": statistics.median(query_ms) if query_ms else None,
                    "p95_query_ms": sorted(query_ms)[math.ceil(.95 * len(query_ms)) - 1] if query_ms else None,
                    "mean_embedding_ms": statistics.mean(embedding_ms) if embedding_ms else None,
                    "mean_rank_ms": statistics.mean(ranking_ms) if ranking_ms else None,
                    "scope": "includes local query embedding and dense ranking; excludes answer generation"},
        "retrieval_metrics_all_holdout": summarize(by_type),
        "retrieval_metrics_fully_covered_only": summarize(eligible),
    }
