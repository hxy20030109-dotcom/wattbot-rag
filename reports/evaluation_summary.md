# Public evaluation summary

This summary reports measured aggregate results only. The detailed local reports remain in the development workspace and are not part of the public release because they contain competition question IDs, question/answer details, or source excerpts.

## Answer generation

- Evaluation: 192-question development split from the training set; results were used during iteration and are not blind validation.
- Pipeline: BM25 V2.1, local `gpt-oss:20b`, sentence-ID evidence selection, exact quotation mapping, and citation validation.
- Official scorer: score 0.697292; 128/192 answer values correct; citation F1 0.736458.
- Mean end-to-end latency: 3.57 seconds; P95: 4.65 seconds.
- Hybrid generation on the same split: score 0.675677; 123/192 correct values; citation F1 0.726042.

## Retrieval

On a separate 51-answerable-question retrieval holdout, dense cosine retrieval measured Recall@5 of 0.9379. BM25+dense RRF measured MRR@5 of 0.9412 and Recall@20 of 1.0000. These metrics score source-document ranking, not end-to-end answer correctness. The split has been used during project iteration.

## Reviewed error categories

Qualitative review of development runs identified numeric scale/unit handling, range formatting, wrong scope among nearby statistics, evidence selection or reasoning despite retrieving a relevant source, abstention calibration, and citation precision/recall as recurring failure modes. No individual examples or category counts are included in this public summary.

## Cost and environment

The measured answer runs used a local Ollama model and did not call a paid inference API. Reported latency is from one local run; electricity and hardware costs were not measured. These figures should not be treated as production benchmarks.
