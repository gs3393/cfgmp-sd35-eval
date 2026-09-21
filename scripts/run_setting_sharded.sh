#!/usr/bin/env bash
# Like run_setting.sh, but splits the prompts over every GPU in CUDA_VISIBLE_DEVICES (one generate.py per GPU),
# then scores once. Seeds depend only on the prompt index, so images equal a single-GPU run given the same batch.
#   usage: CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_setting_sharded.sh <run name> <config> [generate.py args...]
set -uo pipefail
WS=${WS:-$HOME/data/code/cfgmp-sd35-eval-ws}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HARNESS_ROOT=${HARNESS_ROOT:-$WS/RevisitingCFGMethods}
export GENEVAL_ROOT=${GENEVAL_ROOT:-$WS/geneval-bench}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$WS/hf-cache/hub}
name=$1; config=$2; shift 2
out=${OUT_ROOT:-$WS/outputs}/$name
mkdir -p "$out" "$WS/logs"
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
n=${#gpus[@]}

pids=()
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} "$WS/.venv/bin/python" "$REPO/scripts/generate.py" --config "$config" \
    --prompts "$GENEVAL_ROOT/prompts/evaluation_metadata.jsonl" --out "$out" --shard "$i/$n" "$@" \
    > "$WS/logs/$name.shard$i.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "[$name] a generation shard failed, see $WS/logs/$name.shard*.log"; exit 1; }

export CUDA_VISIBLE_DEVICES=${gpus[0]}
"$WS/.venv/bin/python" "$REPO/scripts/geneval_eval.py" "$out" --outfile "$out/results.jsonl" > "$WS/logs/$name.geneval.log" 2>&1
"$WS/.venv/bin/python" "$GENEVAL_ROOT/evaluation/summary_scores.py" "$out/results.jsonl" | tee "$out/summary.txt"
if [ -x "$WS/.venv-pref/bin/python" ]; then
  "$WS/.venv-pref/bin/python" "$REPO/scripts/pref_scores.py" "$out" --outfile "$out/pref_scores.jsonl" > "$WS/logs/$name.pref.log" 2>&1
  tail -1 "$WS/logs/$name.pref.log"
fi
