"""Generate GenEval images for one guidance setting, in the folder layout GenEval's evaluator reads.

    <out>/<prompt index, 5 digits>/metadata.jsonl
    <out>/<prompt index, 5 digits>/samples/<sample index, 5 digits>.png

Every setting uses generator seed `seed + prompt_index`, the same dtype and the same `prepare_latents`,
so image k of prompt i starts from identical noise in every setting (paired comparison).
Resumable: prompts whose samples all exist are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.environ.get("HARNESS_ROOT", str(REPO_ROOT.parent / "RevisitingCFGMethods")))

from pipelines.sd35._patch import SD35_MEDIUM, patch_diffusers_no_bnb  # noqa: E402

patch_diffusers_no_bnb()

from cfgmp_eval.pipeline import CFGMP_METHODS, CFGMPMethodsPipeline, planned_nfe  # noqa: E402


def parse_overrides(pairs):
    return {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<setting>.yaml")
    ap.add_argument("--prompts", required=True, help="GenEval evaluation_metadata.jsonl")
    ap.add_argument("--out", required=True, help="output directory for this setting")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override generation_params")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    ap.add_argument("--stride", type=int, default=1, help="every k-th prompt (pilot across all tasks)")
    ap.add_argument("--shard", default="0/1", help="i/n: handle prompts with index %% n == i")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--model", default=SD35_MEDIUM, help="HF repo id, e.g. stabilityai/stable-diffusion-3.5-large")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config))
    params = dict(conf["generation_params"])
    params.update(parse_overrides(args.set))
    method = params["method"]

    metadata = [json.loads(line) for line in open(args.prompts)]
    shard_i, shard_n = map(int, args.shard.split("/"))
    todo = [i for i in range(len(metadata)) if i % args.stride == 0][: args.limit]
    todo = [i for i in todo if i % shard_n == shard_i]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if method in CFGMP_METHODS:
        nfe = planned_nfe(
            params.get("num_inference_steps", 10), params.get("max_aa_iter", 3),
            params.get("time_threshold", 0.6),
        )
    else:
        nfe = 2 * params["num_inference_steps"]
    run_info = {
        "config": args.config, "generation_params": params, "planned_nfe_per_image": nfe,
        "n_samples": args.n_samples, "seed": args.seed, "dtype": args.dtype, "model": args.model,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / f"run_info.shard{shard_i}of{shard_n}.json").write_text(json.dumps(run_info, indent=2))
    print(json.dumps(run_info, indent=2), flush=True)

    pipe = CFGMPMethodsPipeline.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    done, t0 = 0, time.time()
    for idx in todo:
        sample_dir = out / f"{idx:05d}" / "samples"
        if all((sample_dir / f"{k:05d}.png").is_file() for k in range(args.n_samples)):
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        (out / f"{idx:05d}" / "metadata.jsonl").write_text(json.dumps(metadata[idx]))

        generator = torch.Generator("cuda").manual_seed(args.seed + idx)
        images = pipe(
            metadata[idx]["prompt"], num_images_per_prompt=args.n_samples, generator=generator, **params
        ).images
        if method in CFGMP_METHODS and pipe.last_nfe != nfe:
            print(f"  note: prompt {idx} used {pipe.last_nfe} NFE (planned {nfe})", flush=True)
        for k, image in enumerate(images):
            image.save(sample_dir / f"{k:05d}.png")

        done += 1
        if done % 10 == 0 or done == 1:
            rate = (time.time() - t0) / (done * args.n_samples)
            print(f"[{method}] {done} prompts this run, {rate:.2f} s/image, prompt index {idx}", flush=True)

    print(f"DONE {out} ({done} prompts generated, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
