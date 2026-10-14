"""Small pure-Python BM25 index; deliberately transparent baseline."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("No chunks found. Parse PDFs and chunk pages first.")
        self.documents = documents
        self.tokens = [tokenize(doc.get("text", "")) for doc in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(self.documents)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25":
        with path.open("r", encoding="utf-8") as handle:
            return cls([json.loads(line) for line in handle if line.strip()])

    def search(self, query: str, k: int = 10) -> list[dict]:
        terms = tokenize(query)
        scored = []
        for index, tf in enumerate(self.term_freqs):
            dl = self.lengths[index]
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                    score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, index in scored[:k]:
            row = dict(self.documents[index])
            row["score"] = score
            results.append(row)
        return results

    def search_documents(self, query: str, k: int = 10) -> list[dict]:
        """Rank distinct source documents by their best matching chunk."""
        terms = tokenize(query)
        scored = []
        for index, tf in enumerate(self.term_freqs):
            dl = self.lengths[index]
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                    score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        seen = set()
        for score, index in scored:
            doc = self.documents[index]
            doc_id = str(doc.get("doc_id", "")).casefold()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            row = dict(doc)
            row["score"] = score
            results.append(row)
            if len(results) == k:
                break
        return results

