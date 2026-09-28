#!/usr/bin/env bash
# For hosts that power off when GPU utilisation is low and are kept alive by a dummy-load script.
# Frees the GPUs a job needs, keeps the dummy load on the others, and ALWAYS restores the full dummy load on
# exit (success, failure or signal).
#   usage: scripts/with_keepalive.sh <work gpus, e.g. 0 or 0,1,2,3> -- <command...>
set -uo pipefail
KA_DIR=${KA_DIR:?set KA_DIR to the directory that holds gpu_keepalive.py}
KA_PY=${KA_PY:-/usr/bin/python}
KA_SCRIPT=${KA_SCRIPT:-gpu_keepalive.py}
KA_ARGS=${KA_ARGS:---mem-gb 20 --util 1 --duration 0}

work=$1; shift; [ "$1" = "--" ] && shift
all=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
rest=$(comm -23 <(tr , "\n" <<<"$all" | sort) <(tr , "\n" <<<"$work" | sort) | paste -sd,)

stop_ka() {  # the parent does not reap its per-GPU workers on SIGTERM, so collect them first
  local parents kids
  parents=$(pgrep -f "^[^ ]*python[0-9.]* $KA_SCRIPT" || true)
  [ -z "$parents" ] && return 0
  kids=$(for p in $parents; do pgrep -P "$p"; done)
  kill $parents $kids 2>/dev/null
  for _ in $(seq 20); do
    [ -z "$(for p in $parents $kids; do [ -d /proc/$p ] && echo $p; done)" ] && return 0
    sleep 0.5
  done
  kill -9 $parents $kids 2>/dev/null
}
start_ka() {  # start_ka <gpu list>; empty list = nothing to cover
  [ -z "$1" ] && return 0
  ( cd "$KA_DIR" && setsid nohup "$KA_PY" "$KA_SCRIPT" $KA_ARGS --gpus "$1" >> "$KA_DIR/keepalive.log" 2>&1 < /dev/null & )
}
restore() { stop_ka; start_ka "$all"; echo "[keepalive] $(date -Is) restored on GPUs $all"; }
trap restore EXIT
trap "exit 130" INT TERM HUP

stop_ka
start_ka "$rest"
echo "[keepalive] $(date -Is) work GPUs: $work | dummy load stays on: ${rest:-none}"
CUDA_VISIBLE_DEVICES=$work "$@"
