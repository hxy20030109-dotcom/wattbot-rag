"""Small local Ollama embedding client and resumable corpus index builder."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "nomic-embed-text:v1.5"


def _model_runtime(base_url: str, model: str) -> dict:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    with urllib.request.urlopen(root + "/api/tags", timeout=15) as response:
        tags = json.load(response)
    found = next((item for item in tags.get("models", []) if item.get("name") == model), None)
    if found is None:
        raise ValueError(f"Ollama model is not installed: {model}")
    with urllib.request.urlopen(root + "/api/version", timeout=15) as response:
        version = json.load(response).get("version")
    return {"model": model, "digest": found.get("digest"), "details": found.get("details"),
            "ollama_version": version}


class OllamaEmbedder:
    """Use a persistent local HTTP connection to Ollama's legacy embedding endpoint."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = "http://localhost:11434"):
        parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Embedding endpoint must be an HTTP(S) Ollama URL")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prefix = "/api/embeddings"
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = connection_type(parsed.hostname, parsed.port, timeout=120)

    def embed(self, text: str, task: str) -> np.ndarray:
        if task not in {"search_document", "search_query"}:
            raise ValueError("Nomic embedding task must be search_document or search_query")
        body = json.dumps({"model": self.model, "prompt": f"{task}: {text}"}, ensure_ascii=False)
        try:
            self.connection.request("POST", self.prefix, body=body.encode("utf-8"),
                                    headers={"Content-Type": "application/json"})
            response = self.connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise RuntimeError(f"Ollama embeddings returned HTTP {response.status}: {raw[:500]!r}")
            result = json.loads(raw.decode("utf-8"))
        except Exception:
            self.connection.close()
            raise
        vector = np.asarray(result["embedding"], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Ollama returned an invalid embedding vector")
        return vector

    def close(self) -> None:
        self.connection.close()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_embedding_index(chunks_path: Path, output_path: Path, model: str = DEFAULT_MODEL,
                          base_url: str = "http://localhost:11434", *, resume: bool = False,
                          checkpoint_every: int = 32) -> dict:
    """Embed all chunks with per-document prefixes and resumable disk checkpoints."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    chunks_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    chunks = _read_jsonl(chunks_path)
    if not chunks:
        raise ValueError("Chunk file is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")
    partial_path = output_path.with_suffix(".partial.npy")
    partial_meta_path = output_path.with_suffix(".partial.json")
    runtime = _model_runtime(base_url, model)
    immutable = {"model": model, "model_digest": runtime["digest"],
                 "chunks_sha256": chunks_hash, "chunk_count": len(chunks),
                 "embedding_prefix": "search_document: ", "query_prefix": "search_query: "}
    if output_path.exists() or meta_path.exists():
        if output_path.exists() and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(existing.get(key) == value for key, value in immutable.items()):
                return {"status": "already_built", "output": str(output_path),
                        "chunks": len(chunks), "dimensions": existing["dimensions"],
                        "model_digest": runtime["digest"]}
        raise FileExistsError("Embedding index exists with different settings; choose another output path")

    started = time.perf_counter()
    embedder = OllamaEmbedder(model, base_url)
    try:
        if partial_path.exists() or partial_meta_path.exists():
            if not resume or not (partial_path.exists() and partial_meta_path.exists()):
                raise FileExistsError("Partial embedding build exists; pass --resume or choose a new output")
            progress = json.loads(partial_meta_path.read_text(encoding="utf-8"))
            if progress["immutable"] != immutable:
                raise ValueError("Cannot resume with a different model or chunk snapshot")
            dimensions = progress["dimensions"]
            completed = progress["completed_rows"]
            vectors = np.lib.format.open_memmap(partial_path, mode="r+")
            if vectors.shape != (len(chunks), dimensions):
                raise ValueError("Partial vector matrix has an unexpected shape")
        else:
            first = embedder.embed(chunks[0]["text"], "search_document")
            dimensions = int(first.shape[0])
            vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                                shape=(len(chunks), dimensions))
            norm = float(np.linalg.norm(first))
            if norm == 0:
                raise ValueError("Embedding model returned a zero vector")
            vectors[0] = first / norm
            completed = 1
            vectors.flush()
            progress = {"immutable": immutable, "dimensions": dimensions,
                        "completed_rows": completed, "runtime": runtime}
            partial_meta_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        for row_i in range(completed, len(chunks)):
            vector = embedder.embed(chunks[row_i]["text"], "search_document")
            if vector.shape != (dimensions,):
                raise ValueError(f"Embedding dimension changed at chunk row {row_i}")
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise ValueError(f"Embedding model returned a zero vector at row {row_i}")
            vectors[row_i] = vector / norm
            completed = row_i + 1
            if completed % checkpoint_every == 0 or completed == len(chunks):
                vectors.flush()
                progress = {"immutable": immutable, "dimensions": dimensions,
                            "completed_rows": completed, "runtime": runtime}
                partial_meta_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                print(f"embedded {completed}/{len(chunks)}", flush=True)
    finally:
        embedder.close()

    vectors.flush()
    del vectors
    partial_path.replace(output_path)
    metadata = {
        **immutable, "dimensions": dimensions, "normalized": True,
        "runtime": runtime,
        "chunk_ids": [str(row.get("chunk_id", "")) for row in chunks],
        "doc_ids": [str(row.get("doc_id", "")) for row in chunks],
        "pages": [row.get("page") for row in chunks],
        "index_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_seconds": time.perf_counter() - started,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_meta_path.unlink(missing_ok=True)
    return {"status": "built", "output": str(output_path), "chunks": len(chunks),
            "dimensions": dimensions, "model_digest": runtime["digest"],
            "build_seconds": metadata["build_seconds"]}
