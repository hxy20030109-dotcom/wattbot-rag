"""Protect evidence-validation boundaries without contacting a model or network."""

import unittest
import json
import os
from unittest.mock import patch
from unittest.mock import MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from wattbot.generation import _api_call, answer_question
from wattbot.dense_retrieval import DenseIndex
from wattbot.embeddings import OllamaEmbedder
from wattbot.generation_v2 import RESPONSE_SCHEMA_V2, _call_model, _sentence_units, answer_question_v2
from wattbot.normalize import normalize_answer_value, normalize_predictions
from wattbot.retrieval import BM25


class GroundingBoundaries(unittest.TestCase):
    def setUp(self):
        self.index = BM25([{"doc_id": "source", "title": "Report", "page": 3,
                            "text": "The reported emissions were 42 metric tons."}])
        self.metadata = {"source": {"url": "https://example.org/report.pdf"}}
        self.valid = {"answerable": True, "answer": "42 metric tons", "answer_value": "42",
                      "answer_unit": "metric tons", "explanation": "The report states 42 metric tons.",
                      "evidence": [{"doc_id": "source", "quote": "emissions were 42 metric tons."}]}

    def run_answer(self, response):
        trace = {}
        with patch("wattbot.generation._api_call", return_value=response) as endpoint:
            answer, _ = answer_question("What emissions were reported?", self.index, self.metadata,
                                         answer_unit="metric tons", trace=trace)
        return answer, trace, endpoint

    def test_fabricated_quote_cannot_pass_with_a_valid_document_id(self):
        self.valid["evidence"][0]["quote"] = "emissions were 99 metric tons."
        answer, trace, _ = self.run_answer(self.valid)
        self.assertEqual(answer["answer_value"], "is_blank")
        self.assertEqual(answer["ref_id"], "is_blank")
        self.assertEqual(trace["decision"], "quote_not_in_retrieved_passage")

    def test_real_quote_from_an_unretrieved_document_is_rejected(self):
        self.metadata["other"] = {"url": "https://example.org/other.pdf"}
        self.valid["evidence"][0]["doc_id"] = "other"
        answer, _, _ = self.run_answer(self.valid)
        self.assertEqual(answer["supporting_materials"], "is_blank")

    def test_string_false_is_not_treated_as_true(self):
        self.valid["answerable"] = "false"
        answer, trace, _ = self.run_answer(self.valid)
        self.assertEqual(answer["answer_value"], "is_blank")
        self.assertEqual(trace["decision"], "invalid_answerable_type")

    def test_benchmark_unit_is_forwarded_without_answer_labels(self):
        answer, _, endpoint = self.run_answer(self.valid)
        self.assertEqual(endpoint.call_args.kwargs["answer_unit"], "metric tons")
        self.assertIn('source p.3:', answer["supporting_materials"])

    def test_model_refusal_clears_all_evidence_fields(self):
        self.valid["answerable"] = False
        answer, _, _ = self.run_answer(self.valid)
        for field in ("answer_value", "ref_id", "ref_url", "supporting_materials"):
            self.assertEqual(answer[field], "is_blank")

    def test_case_variant_citation_maps_to_canonical_id(self):
        self.valid["evidence"][0]["doc_id"] = "SOURCE"
        answer, trace, _ = self.run_answer(self.valid)
        self.assertEqual(trace["decision"], "accepted")
        self.assertEqual(answer["ref_id"], '["source"]')

    def test_whitespace_repair_outputs_the_actual_source_substring(self):
        self.valid["evidence"][0]["quote"] = "emissions were 42\u202fmetric tons."
        answer, trace, _ = self.run_answer(self.valid)
        self.assertEqual(trace["decision"], "accepted")
        self.assertIn('"emissions were 42 metric tons."', answer["supporting_materials"])
        self.assertEqual(trace["quote_whitespace_restorations"], 1)

    def test_ollama_request_sets_context_and_separate_output_budget(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "model": "test-model", "message": {"content": json.dumps(self.valid)},
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}).encode()
        env = {"WATTBOT_LLM_BASE_URL": "http://localhost:11434/v1", "WATTBOT_LLM_MODEL": "test-model",
               "WATTBOT_LLM_BACKEND": "ollama", "WATTBOT_LLM_CONTEXT": "8192", "WATTBOT_LLM_MAX_TOKENS": "2048"}
        trace = {}
        with patch.dict(os.environ, env), patch("urllib.request.urlopen", return_value=response) as request:
            _api_call("Question", self.index.documents, answer_unit="metric tons", trace=trace)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertTrue(request.call_args.args[0].full_url.endswith("/api/chat"))
        self.assertEqual(trace["llm_response"]["usage"]["total_tokens"], 30)

    def test_submission_boolean_values_match_numeric_benchmark_encoding(self):
        for value, expected in (("TRUE", "1"), ("True", "1"), ("FALSE", "0"), ("false", "0")):
            with self.subTest(value=value):
                self.assertEqual(normalize_answer_value(value), expected)

    def test_submission_normalization_changes_only_answer_value(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            source = Path(directory) / "input.csv"
            destination = Path(directory) / "output.csv"
            source.write_text(
                'id,answer_value,explanation\nq1,FALSE,"The answer is FALSE."\n', encoding="utf-8"
            )
            summary = normalize_predictions(source, destination)
            self.assertEqual(summary["rows_changed"], 1)
            self.assertEqual(
                destination.read_text(encoding="utf-8-sig"),
                'id,answer_value,explanation\nq1,0,The answer is FALSE.\n',
            )

    def test_v2_sentence_ids_recover_exact_original_spans(self):
        units = _sentence_units([{"chunk_id": "c1", "doc_id": "source", "page": 3,
                                  "text": "The answer is 42. The next sentence is unrelated."}],
                                question="answer")
        self.assertEqual(units[0]["evidence_id"], "E001")
        self.assertEqual(units[0]["text"], "The answer is 42.")

    def test_v2_citations_are_derived_from_selected_retrieved_sentence_ids(self):
        response = {"answerable": True, "answer": "42", "answer_value": "42",
                    "answer_unit": "metric tons", "explanation": "The reported value is 42.",
                    "evidence_ids": ["E001"]}
        index = BM25([{"doc_id": "source", "title": "Report", "page": 3,
                       "chunk_id": "c1", "text": "The answer is 42. The next sentence is unrelated."}])
        trace = {}
        with patch("wattbot.generation_v2._call_model", return_value=response):
            answer, _ = answer_question_v2("What is the answer?", index, self.metadata,
                                           answer_unit="metric tons", trace=trace)
        self.assertEqual(trace["decision"], "accepted")
        self.assertIn('"The answer is 42."', answer["supporting_materials"])
        self.assertNotIn("unrelated", answer["supporting_materials"])

    def test_v2_normalized_boolean_is_written_into_explanation(self):
        response = {"answerable": True, "answer": "False", "answer_value": "False",
                    "answer_unit": "is_blank", "explanation": "The evidence contradicts the claim.",
                    "evidence_ids": ["E001"]}
        index = BM25([{"doc_id": "source", "title": "Report", "page": 3,
                       "chunk_id": "c1", "text": "The measured ratio was 1.4."}])
        with patch("wattbot.generation_v2._call_model", return_value=response):
            answer, _ = answer_question_v2("What measured ratio was reported?", index, self.metadata)
        self.assertEqual(answer["answer_value"], "0")
        self.assertIn("Normalized answer value: 0", answer["explanation"])

    def test_v2_exact_answer_value_can_be_stated_deterministically_in_explanation(self):
        response = {"answerable": True, "answer": "42", "answer_value": "42",
                    "answer_unit": "metric tons", "explanation": "The report supports this value.",
                    "evidence_ids": ["E001"]}
        index = BM25([{"doc_id": "source", "title": "Report", "page": 3,
                       "chunk_id": "c1", "text": "The answer is 42."}])
        with patch("wattbot.generation_v2._call_model", return_value=response):
            answer, _ = answer_question_v2("What is the answer?", index, self.metadata)
        self.assertIn("Normalized answer value: 42", answer["explanation"])

    def test_v2_ollama_request_uses_required_json_schema(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "model": "test-model", "message": {"content": json.dumps({"answerable": False})},
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}).encode()
        env = {"WATTBOT_LLM_BASE_URL": "http://localhost:11434/v1", "WATTBOT_LLM_MODEL": "test-model",
               "WATTBOT_LLM_BACKEND": "ollama"}
        with patch.dict(os.environ, env), patch("urllib.request.urlopen", return_value=response) as request:
            _call_model("Question", [], "", {})
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["format"], RESPONSE_SCHEMA_V2)

    def test_nomic_embedder_adds_asymmetric_retrieval_prefixes(self):
        class FakeResponse:
            status = 200

            @staticmethod
            def read():
                return b'{"embedding":[3,4]}'

        class FakeConnection:
            def __init__(self, *args, **kwargs):
                self.payload = None

            def request(self, method, path, body, headers):
                self.payload = (method, path, json.loads(body))

            @staticmethod
            def getresponse():
                return FakeResponse()

            @staticmethod
            def close():
                pass

        with patch("wattbot.embeddings.http.client.HTTPConnection", FakeConnection):
            embedder = OllamaEmbedder("local-model")
            query_vector = embedder.embed("how much water?", "search_query")
            document_vector = embedder.embed("water usage was 42", "search_document")
        self.assertEqual(query_vector.tolist(), [3.0, 4.0])
        self.assertEqual(document_vector.tolist(), [3.0, 4.0])
        self.assertEqual(embedder.connection.payload[2]["prompt"], "search_document: water usage was 42")

    def test_dense_document_ranking_keeps_best_chunk_per_source(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            vectors_path = root / "vectors.npy"
            chunks_path = root / "chunks.jsonl"
            np.save(vectors_path, np.asarray([[1, 0], [.9, .1], [0, 1]], dtype=np.float32))
            vectors_path.with_suffix(".meta.json").write_text(json.dumps({"chunk_count": 3, "model": "fake"}),
                                                                encoding="utf-8")
            chunks_path.write_text(
                '{"doc_id":"a","chunk_id":"a1","text":"best"}\n'
                '{"doc_id":"a","chunk_id":"a2","text":"second"}\n'
                '{"doc_id":"b","chunk_id":"b1","text":"other"}\n', encoding="utf-8")

            class FakeEmbedder:
                def __init__(self, *args, **kwargs):
                    pass

                @staticmethod
                def embed(query, task):
                    return np.asarray([1, 0], dtype=np.float32)

                @staticmethod
                def close():
                    pass

            with patch("wattbot.dense_retrieval.OllamaEmbedder", FakeEmbedder):
                index = DenseIndex(vectors_path, chunks_path)
                results, _ = index.search_documents("query", 2)
                index.close()
            self.assertEqual([row["doc_id"] for row in results], ["a", "b"])
            self.assertEqual([row["chunk_id"] for row in results], ["a1", "b1"])


if __name__ == "__main__":
    unittest.main()
