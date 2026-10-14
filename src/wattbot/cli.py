"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wattbot.audit import audit
from wattbot.chunk import chunk_pages
from wattbot.download import download_selected
from wattbot.evaluate import evaluate
from wattbot.dense_retrieval import evaluate_dense
from wattbot.embeddings import DEFAULT_MODEL, build_embedding_index
from wattbot.hybrid_retrieval import evaluate_hybrid
from wattbot.mirror import fetch_arxiv_mirror
from wattbot.normalize import normalize_predictions
from wattbot.pdf import extract_pages
from wattbot.predict import holdout_predictions, predict
from wattbot.predict_v2 import predict_v2
from wattbot.retrieval import BM25


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="wattbot", description="WattBot RAG data and retrieval tools")
    subs = root.add_subparsers(dest="command", required=True)

    p = subs.add_parser("audit", help="audit local competition CSVs and hashes")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))

    p = subs.add_parser("download-pdf", help="download explicitly selected PDF IDs")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    p.add_argument("--id", dest="doc_ids", action="append", required=True)
    p.add_argument("--delay", type=float, default=4.0)

    p = subs.add_parser("download-arxiv-mirror", help="fetch pinned arXiv versions from the sanctioned Cornell GCS feed")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    p.add_argument("--limit", type=int)
    p.add_argument("--version-map", type=Path)

    p = subs.add_parser("parse-pdfs", help="extract cached PDFs page by page")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    p.add_argument("--out", type=Path, default=Path("data/processed/pages.jsonl"))

    p = subs.add_parser("chunk", help="make page-aware chunks from extracted text")
    p.add_argument("--pages", type=Path, default=Path("data/processed/pages.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--size", type=int, default=1800)
    p.add_argument("--overlap", type=int, default=250)

    p = subs.add_parser("retrieve", help="retrieve evidence candidates with BM25")
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=10)

    p = subs.add_parser("evaluate-retrieval", help="evaluate held-out citation retrieval")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])

    p = subs.add_parser("evaluate-hybrid", help="measure BM25+dense RRF retrieval on the fixed training holdout")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--vectors", type=Path, default=Path("data/indexes/nomic-embed-text-v1.5.npy"))
    p.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--rrf-k", type=int, default=60)

    p = subs.add_parser("build-dense-index", help="build a resumable local embedding index with Ollama")
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/indexes/nomic-embed-text-v1.5.npy"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=32)

    p = subs.add_parser("evaluate-dense", help="measure dense document retrieval on the fixed training holdout")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--vectors", type=Path, default=Path("data/indexes/nomic-embed-text-v1.5.npy"))
    p.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])

    p = subs.add_parser("predict", help="generate evidence-cited CSV with a configured LLM")
    p.add_argument("--input", type=Path, default=Path("data/competition/test_Q.csv"))
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--resume", action="store_true")

    p = subs.add_parser("predict-v2", help="generate answers with programmatically extracted sentence citations")
    p.add_argument("--input", type=Path, default=Path("data/competition/test_Q.csv"))
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/predictions_v2.csv"))
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--retrieval", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--vectors", type=Path, default=Path("data/indexes/nomic-embed-text-v1.5.npy"))
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--resume", action="store_true")

    p = subs.add_parser("evaluate-answers", help="score generated held-out answers with Score.py")
    p.add_argument("--data-dir", type=Path, default=Path("data/competition"))
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/holdout"))
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--resume", action="store_true")

    p = subs.add_parser("normalize-submission", help="canonicalize answer values in a copied prediction CSV")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    if args.command == "audit":
        raise SystemExit(audit(args.data_dir))
    if args.command == "download-pdf":
        download_selected(args.data_dir, args.pdf_dir, args.doc_ids, args.delay)
    elif args.command == "download-arxiv-mirror":
        print(json.dumps(fetch_arxiv_mirror(args.data_dir, args.pdf_dir, args.limit, args.version_map), indent=2))
    elif args.command == "parse-pdfs":
        print(json.dumps(extract_pages(args.data_dir, args.pdf_dir, args.out), indent=2))
    elif args.command == "chunk":
        print(f"chunks={chunk_pages(args.pages, args.out, args.size, args.overlap)}")
    elif args.command == "retrieve":
        for row in BM25.from_jsonl(args.chunks).search(args.query, args.k):
            print(json.dumps(row, ensure_ascii=False))
    elif args.command == "evaluate-retrieval":
        print(json.dumps(evaluate(args.data_dir, args.chunks, args.k), indent=2))
    elif args.command == "build-dense-index":
        print(json.dumps(build_embedding_index(args.chunks, args.output, args.model, args.base_url,
                                               resume=args.resume, checkpoint_every=args.checkpoint_every), indent=2))
    elif args.command == "evaluate-dense":
        print(json.dumps(evaluate_dense(args.data_dir, args.chunks, args.vectors, args.k), indent=2))
    elif args.command == "evaluate-hybrid":
        print(json.dumps(evaluate_hybrid(args.data_dir, args.chunks, args.vectors, args.k, args.rrf_k), indent=2))
    elif args.command == "predict":
        print(json.dumps(predict(args.input, args.chunks, args.data_dir / "metadata.csv", args.output, args.top_k, resume=args.resume), indent=2))
    elif args.command == "predict-v2":
        print(json.dumps(predict_v2(args.input, args.chunks, args.data_dir / "metadata.csv", args.output,
                                    args.top_k, resume=args.resume, retrieval=args.retrieval,
                                    vectors_path=args.vectors, rrf_k=args.rrf_k), indent=2))
    elif args.command == "evaluate-answers":
        import subprocess
        import sys

        solution_path, prediction_path = holdout_predictions(
            args.data_dir, args.chunks, args.output_dir, args.top_k, resume=args.resume
        )
        subprocess.run(
            [sys.executable, str(args.data_dir / "Score.py"), str(solution_path), str(prediction_path)],
            check=True,
        )
    elif args.command == "normalize-submission":
        print(json.dumps(normalize_predictions(args.input, args.output), ensure_ascii=False, indent=2))


