#!/usr/bin/env bash
# SD3.5-Large queue (paper condition: w=4, ~60 NFE). Run under with_keepalive.sh with the work GPUs in
# CUDA_VISIBLE_DEVICES. Output names start with L_. Resumable: finished settings are skipped, generation resumes.
set -uo pipefail
WS=${WS:-$HOME/data/code/cfgmp-sd35-eval-ws}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT_ROOT:-$WS/outputs}
MODEL=stabilityai/stable-diffusion-3.5-large

run() {  # run <name> <config> [generate.py args...]
  local name=$1; shift
  if [ -f "$OUT/$name/summary.txt" ] && [ -f "$OUT/$name/pref_scores.jsonl" ]; then echo "[queue] $name done, skip"; return; fi
  echo "[queue] $(date -Is) start $name"
  "$REPO/scripts/run_setting_sharded.sh" "$name" "$@" --model $MODEL > "$WS/logs/$name.log" 2>&1 || echo "[queue] $name FAILED (see logs/$name*.log)"
  echo "[queue] $(date -Is) end $name: $(grep Overall "$OUT/$name/summary.txt" 2>/dev/null) | $(tail -1 "$WS/logs/$name.pref.log" 2>/dev/null)"
}

run L_cfg          "$REPO/configs/cfg.yaml"
run L_cfgmpp       "$REPO/configs/cfgmpp.yaml"
run L_cfgmp        "$REPO/configs/cfgmp.yaml"
# CFG-Zero* at w=4 as in the CFG-MP paper (the harness value 3.5 was tuned on Medium, not Large)
run L_cfg0s        "$REPO/configs/cfg0s.yaml" --set guidance_scale=4.0
run L_cfgmpp_s20k1 "$REPO/configs/cfgmpp_s20k1.yaml"
for w in 3 5 7 9 12; do
  for s in cfg cfgmpp; do run "L_sweep_${s}_w$w" "$REPO/configs/$s.yaml" --n-samples 1 --set guidance_scale=$w; done
done
echo "[queue] ALL DONE $(date -Is)"
