#!/usr/bin/env bash
# Sequential queue on one GPU: main settings (4 images/prompt) with GenEval + preference scores,
# then the guidance-scale sweep (1 image/prompt). Resumable: finished steps are skipped.
set -uo pipefail
# Workspace = the directory that contains this repo clone (<ws>/cfgmp-sd35-eval). Override with CFGMP_WS, never the
# generic WS: a tmux server on a shared host can carry another project's WS in its global environment.
WS=${CFGMP_WS:-$(cd "$(dirname "$0")/../.." && pwd)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT_ROOT:-$WS/outputs}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$WS/hf-cache/hub}
mkdir -p "$WS/logs"

run() {  # run <name> <config> [generate.py args...]
  local name=$1
  if [ -f "$OUT/$name/summary.txt" ]; then echo "[queue] $name: GenEval done, skip"; else
    echo "[queue] $(date -Is) start $name"
    "$REPO/scripts/run_setting.sh" "$@" > "$WS/logs/$name.log" 2>&1 || echo "[queue] $name FAILED (see logs/$name.log)"
  fi
  if [ -f "$OUT/$name/summary.txt" ] && [ ! -f "$OUT/$name/pref_scores.jsonl" ]; then
    "$WS/.venv-pref/bin/python" "$REPO/scripts/pref_scores.py" "$OUT/$name" --outfile "$OUT/$name/pref_scores.jsonl" \
      > "$WS/logs/$name.pref.log" 2>&1 || echo "[queue] $name pref scores FAILED"
  fi
  echo "[queue] $(date -Is) end $name: $(tail -1 "$OUT/$name/summary.txt" 2>/dev/null)"
}

for s in cfg cfgmpp cfgmpp_s20k1 cfg0s cfgmp apg cfgmpp_s20k3 cfgmp_s10k1; do run "$s" "$REPO/configs/$s.yaml"; done
for w in 3 5 7 9 12; do
  for s in cfg cfgmpp; do run "sweep_${s}_w$w" "$REPO/configs/$s.yaml" --n-samples 1 --set guidance_scale=$w; done
done
echo "[queue] ALL DONE $(date -Is)"
