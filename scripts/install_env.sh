#!/usr/bin/env bash
# harness setup_env.py의 deps + benchmarks(geneval) 단계를 같은 핀으로 수행한다.
# (setup_env.py deps는 공개 저장소에 없는 cfgbench를 editable 설치하려다 실패하므로 그 한 줄만 뺀다.)
set -euxo pipefail
W=${WS:-$HOME/data/code/cfgmp-sd35-eval-ws}
H=$W/RevisitingCFGMethods
export GENEVAL_ROOT=$W/geneval-bench
export PIP_CACHE_DIR=$W/.pip-cache
export TMPDIR=$W/.tmp; mkdir -p $TMPDIR
python3 -m venv $W/.venv
source $W/.venv/bin/activate
PIP="python -m pip install --prefer-binary -c $H/requirements/constraints.txt"
$PIP --upgrade pip setuptools wheel
$PIP -r $H/requirements/torch.txt
$PIP -r $H/requirements/core.txt
cd $H && python setup_env.py benchmarks --only geneval
python - <<PY
import torch, diffusers, mmdet, mmcv
print("torch", torch.__version__, torch.cuda.is_available(), "diffusers", diffusers.__version__, "mmdet", mmdet.__version__, "mmcv", mmcv.__version__)
from mmcv.ops import nms; print("mmcv ops ok")
PY
# GenEval/harness fetch the mmdet v2.0 checkpoint, whose decoder keys do not load into mmdet 3.x (detector finds nothing).
# Use OpenMMLab's v3.0 re-keyed file of the same run (20220504_001756), saved under the name GenEval expects.
M=$GENEVAL_ROOT/models/mmdet3; mkdir -p $M
[ -f $M/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth ] || curl -sfL -o $M/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth   https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth
python -m pip install -q pytest
echo INSTALL_DONE
