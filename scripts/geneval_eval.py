"""Run GenEval's own `evaluation/evaluate_images.py` unchanged on an mmdet 3.x install.

GenEval was written for mmdet 2.x, where `inference_detector` returns per-class lists
`(bbox_results, segm_results)`. mmdet 3.x returns a `DetDataSample`. This wrapper converts the
3.x output back to the 2.x layout and then executes the original script, so every threshold and
rule of the benchmark stays GenEval's.

    python scripts/geneval_eval.py <imagedir> --outfile <imagedir>/results.jsonl
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np

GENEVAL_ROOT = Path(os.environ["GENEVAL_ROOT"])
MM_CONFIG = GENEVAL_ROOT / "mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"

import mmdet.apis  # noqa: E402

_inference_detector_v3 = mmdet.apis.inference_detector


def inference_detector_v2_layout(model, img):
    pred = _inference_detector_v3(model, img).pred_instances
    boxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    masks = pred.masks.cpu().numpy() if "masks" in pred else None
    num_classes = len(model.dataset_meta["classes"])
    bbox_results, segm_results = [], []
    for c in range(num_classes):
        keep = labels == c
        bbox_results.append(np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32).reshape(-1, 5))
        segm_results.append([] if masks is None else list(masks[keep]))
    return bbox_results, segm_results


mmdet.apis.inference_detector = inference_detector_v2_layout

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--model-config" not in argv:
        argv += ["--model-config", str(MM_CONFIG)]
    if "--model-path" not in argv:
        # models/mmdet3 holds OpenMMLab's v3.0 re-keyed checkpoint of the same training run. The v2.0 file that
        # GenEval (and the harness installer) download loads into mmdet 3.x with its decoder weights silently
        # unmatched, and then detects nothing.
        argv += ["--model-path", str(GENEVAL_ROOT / "models" / "mmdet3")]
    script = GENEVAL_ROOT / "evaluation" / "evaluate_images.py"
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")
