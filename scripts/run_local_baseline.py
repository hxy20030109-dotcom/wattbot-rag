"""Run a frozen, digest-checked local baseline, then the shipped scorer."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from wattbot.predict import holdout_predictions

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=root / "configs/bm25_local_v1.json")
parser.add_argument("--output-dir", type=Path, default=root / "outputs/holdout_bm25_v1")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
config = json.loads(args.config.read_text(encoding="utf-8"))
for relative, expected in config["frozen_files_sha256"].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Frozen file changed: {relative}; create a new experiment configuration")
settings = config["inference"]
api_root = settings["base_url"].removesuffix("/v1")
with urllib.request.urlopen(api_root + "/api/tags", timeout=15) as response:
    tags = json.load(response)
model = next((item for item in tags["models"] if item["name"] == settings["model"]), None)
if not model or model["digest"] != settings["model_digest"]:
    raise ValueError("Local model digest does not match the frozen baseline")
for variable, value in {"WATTBOT_LLM_BASE_URL": settings["base_url"], "WATTBOT_LLM_MODEL": settings["model"],
                        "WATTBOT_LLM_BACKEND": settings["backend"], "WATTBOT_LLM_CONTEXT": settings["context_length"],
                        "WATTBOT_LLM_MAX_TOKENS": settings["max_tokens"], "WATTBOT_LLM_THINK": settings["thinking"]}.items():
    os.environ[variable] = str(value)
args.output_dir.mkdir(parents=True, exist_ok=True)
snapshot = args.output_dir / "frozen_configuration.json"
if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != config:
    raise ValueError("Output directory belongs to a different frozen experiment")
snapshot.write_text(json.dumps(config, indent=2), encoding="utf-8")
solution, prediction = holdout_predictions(root / "data/competition", root / "data/processed/chunks.jsonl",
                                           args.output_dir, settings["top_k"], resume=args.resume)
subprocess.run([sys.executable, str(root / "scripts/score_predictions.py"),
                "--solution", str(solution), "--predictions", str(prediction)], check=True)
