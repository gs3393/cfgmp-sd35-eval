#!/usr/bin/env bash
# HPSv2 + ImageReward scorers in their own venv (their pins conflict with the generation stack).
set -euxo pipefail
# Workspace = the directory that contains this repo clone (<ws>/cfgmp-sd35-eval). Override with CFGMP_WS, never the
# generic WS: a tmux server on a shared host can carry another project's WS in its global environment.
W=${CFGMP_WS:-$(cd "$(dirname "$0")/../.." && pwd)}
export PIP_CACHE_DIR=$W/.pip-cache TMPDIR=$W/.tmp
# The pins (torch 2.5.1+cu121, mmcv 2.2.0 wheel) are validated on Python 3.10. Where the system python differs,
# let uv fetch 3.10; its interpreter and cache stay under $W so they survive on hosts with an ephemeral home.
if python3 -c "import sys; sys.exit(sys.version_info[:2] != (3, 10))"; then
  python3 -m venv $W/.venv-pref
else
  export UV_PYTHON_INSTALL_DIR=$W/.uv-python UV_CACHE_DIR=$W/.uv-cache
  uv venv --python 3.10 --seed $W/.venv-pref
fi
source $W/.venv-pref/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "numpy<2"
python -m pip install --prefer-binary hpsv2 image-reward "transformers==4.45.2" "huggingface_hub<0.26" fairscale
python -m pip install --prefer-binary "git+https://github.com/openai/CLIP.git"
# Two defects in the hpsv2 wheel: an unused `from turtle import forward` (needs tkinter, absent on servers)
# and a missing BPE vocab file (identical to the one shipped with openai/CLIP).
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
sed -i "/from turtle import/d" $SP/hpsv2/src/open_clip/factory.py
cp $SP/clip/bpe_simple_vocab_16e6.txt.gz $SP/hpsv2/src/open_clip/
python - <<PY
import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)
import hpsv2, ImageReward; print("imports ok")
PY
echo PREF_INSTALL_DONE
