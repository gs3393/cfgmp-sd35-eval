#!/usr/bin/env bash
# Generate + GenEval-score one setting.  usage: scripts/run_setting.sh <run name> <config> [generate.py args...]
set -euo pipefail
WS=${WS:-$HOME/data/code/cfgmp-sd35-eval-ws}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HARNESS_ROOT=${HARNESS_ROOT:-$WS/RevisitingCFGMethods}
export GENEVAL_ROOT=${GENEVAL_ROOT:-$WS/geneval-bench}
# cache on /mnt/data (root disk is nearly full); HF_HOME stays default so the login token is found
export HF_HUB_CACHE=${HF_HUB_CACHE:-$WS/hf-cache/hub}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
source "$WS/.venv/bin/activate"
name=$1; config=$2; shift 2
out=${OUT_ROOT:-$WS/outputs}/$name
python "$REPO/scripts/generate.py" --config "$config" --prompts "$GENEVAL_ROOT/prompts/evaluation_metadata.jsonl" --out "$out" "$@"
python "$REPO/scripts/geneval_eval.py" "$out" --outfile "$out/results.jsonl"
python "$GENEVAL_ROOT/evaluation/summary_scores.py" "$out/results.jsonl" | tee "$out/summary.txt"
