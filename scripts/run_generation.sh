#!/usr/bin/env bash
# Closed-book vs RAG answer quality with a local 1.5B model on CPU (~30-60 min).
# Retrieval config chosen from results/retrieval.json: best Hit@5 / MRR = dense retrieval, 256-word chunks.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8

$PY -m src.evaluate_generation --name closed-book  --mode closed-book --model "$MODEL"
$PY -m src.evaluate_generation --name rag-dense-256 --mode rag --model "$MODEL" --method dense --chunk-size 256 --k 4
$PY -m src.report
echo "DONE"
