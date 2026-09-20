"""Score an existing GenEval image folder with HPSv2 (v2.0) and ImageReward-v1.0.

Writes one row per image in the same shape as GenEval's results.jsonl (`filename`, `tag`, `prompt`)
plus `hpsv2` and `image_reward`, so `bootstrap_report.py --key hpsv2` works on it unchanged.
Runs in the separate scorer venv (.venv-pref).

    python scripts/pref_scores.py <imagedir> --outfile <imagedir>/pref_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image


def load_hpsv2(device):
    import huggingface_hub
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14", "laion2B-s32B-b79K", precision="amp", device=device, jit=False, force_quick_gelu=False,
        force_custom_text=False, force_patch_dropout=False, force_image_size=None, pretrained_image=False,
        image_mean=None, image_std=None, light_augmentation=True, aug_cfg={}, output_dict=True,
        with_score_predictor=False, with_region_predictor=False,
    )
    checkpoint = huggingface_hub.hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    return model.to(device).eval(), preprocess, get_tokenizer("ViT-H-14")


@torch.no_grad()
def hpsv2_scores(model, preprocess, tokenizer, paths, prompt, device):
    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
    text = tokenizer([prompt]).to(device)
    with torch.cuda.amp.autocast():
        out = model(images, text)
        logits = out["image_features"] @ out["text_features"].T
    return [100.0 * float(x) for x in logits[:, 0]]  # x100, the scale used in the CFG-MP tables


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagedir")
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()
    device = "cuda"

    import ImageReward as RM

    hps_model, hps_preprocess, hps_tokenizer = load_hpsv2(device)
    ir_model = RM.load("ImageReward-v1.0", device=device)

    rows = []
    for folder in sorted(p for p in Path(args.imagedir).iterdir() if p.is_dir() and p.name.isdigit()):
        metadata = json.loads((folder / "metadata.jsonl").read_text())
        paths = sorted(p for p in (folder / "samples").iterdir() if re.fullmatch(r"\d+\.png", p.name))
        if not paths:
            continue
        hps = hpsv2_scores(hps_model, hps_preprocess, hps_tokenizer, paths, metadata["prompt"], device)
        ir = ir_model.score(metadata["prompt"], [str(p) for p in paths])
        ir = [ir] if isinstance(ir, float) else list(ir)
        for path, h, r in zip(paths, hps, ir):
            rows.append({"filename": str(path), "tag": metadata["tag"], "prompt": metadata["prompt"],
                         "hpsv2": h, "image_reward": float(r)})

    with open(args.outfile, "w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")
    n = max(len(rows), 1)
    print(f"{len(rows)} images  HPSv2 {sum(r['hpsv2'] for r in rows) / n:.2f}  "
          f"ImageReward {sum(r['image_reward'] for r in rows) / n:.3f}")


if __name__ == "__main__":
    main()
