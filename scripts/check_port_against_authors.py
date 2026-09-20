"""Run the authors own CFG-MP_SD pipeline and this repo port from the same seed and compare the pixels.

    python scripts/check_port_against_authors.py --out <dir>      (needs CFGMP_ROOT, HARNESS_ROOT)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.environ["HARNESS_ROOT"])
sys.path.insert(0, str(Path(os.environ["CFGMP_ROOT"]) / "CFG-MP_SD"))

from pipelines.sd35._patch import SD35_MEDIUM, patch_diffusers_no_bnb  # noqa: E402

patch_diffusers_no_bnb()

from utils_SD import CFGMPScheduler, CFGMPSD3Pipeline  # noqa: E402  (authors, unmodified)

from cfgmp_eval.pipeline import CFGMPMethodsPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a photo of a cow")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mine = CFGMPMethodsPipeline.from_pretrained(SD35_MEDIUM, torch_dtype=torch.bfloat16).to("cuda")
    components = {**mine.components, "scheduler": CFGMPScheduler(num_train_timesteps=1000, shift=3.0)}
    theirs = CFGMPSD3Pipeline(**components)
    common = dict(num_inference_steps=10, guidance_scale=4.0, max_aa_iter=3, aa_window_size=1,
                  aa_damping=1.0, time_threshold=0.6)

    for use_aa, method in ((True, "cfgmpp"), (False, "cfgmp")):
        gen = torch.Generator("cuda").manual_seed(args.seed)
        a = theirs(prompt=args.prompt, height=1024, width=1024, use_aa=use_aa, generator=gen,
                   output_type="pt", **common)
        gen = torch.Generator("cuda").manual_seed(args.seed)
        b = mine(args.prompt, method=method, generator=gen, output_type="pt", **common).images
        a, b = a.float().cpu(), b.float().cpu()
        diff = (a - b).abs()
        print(f"{method}: max abs pixel diff {diff.max():.5f}, mean {diff.mean():.6f} (range 0..1)")
        for name, t in (("authors", a), ("port", b)):
            mine.image_processor.pt_to_numpy(t)
            mine.image_processor.numpy_to_pil(mine.image_processor.pt_to_numpy(t))[0].save(out / f"{method}_{name}.png")


if __name__ == "__main__":
    main()
