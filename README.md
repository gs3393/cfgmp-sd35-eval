# cfgmp-sd35-eval

Code, per-image scores and bootstrap tables for re-measuring **CFG-MP / CFG-MP+** (Cai, Liu, Su, Wang, *Improving Classifier-Free Guidance of Flow Matching via Manifold Projection*, ICML 2026, [arXiv:2601.21892](https://arxiv.org/abs/2601.21892)) on Stable Diffusion 3.5 Medium and Large, using the paired prompt-bootstrap protocol of *Revisiting Classifier-Free Guidance Methods in Latent Diffusion Models* ([arXiv:2608.16786](https://arxiv.org/abs/2608.16786)).

Every setting generates the same 553 GenEval prompts from the same initial noise (4 images per prompt for the main settings, 1 for the guidance-scale sweep), at 1024² in bfloat16, and scores the same images with GenEval, HPSv2 (v2.0) and ImageReward v1.0. Differences are paired at the image level; intervals are 95% percentile intervals of 10,000 paired bootstrap replicates that resample prompts within each GenEval task. The write-up is a note on [gs3393.github.io](https://gs3393.github.io) (link added when published).

## What was measured

| Setting | Method | Steps | Projection rounds K | Anderson | NFE |
|---|---|---|---|---|---|
| `cfg` | CFG, w = 4 | 30 | — | — | 60 |
| `cfg0s` | CFG-Zero* (Medium w = 3.5 as tuned by the harness; Large w = 4) | 30 | — | — | 60 |
| `apg` | APG, harness-tuned (Medium only) | 30 | — | — | 60 |
| `cfgmp` | CFG-MP (plain fixed-point iteration), `demo_SD.py` defaults | 10 | 3 | no | 62 |
| `cfgmpp` | CFG-MP+ (Anderson AA(1,1)), `demo_SD.py` defaults | 10 | 3 | yes | 62 |
| `cfgmpp_s20k1` | plain iteration — with K = 1 the Anderson branch never activates | 20 | 1 | inactive | 68 |
| `cfgmpp_s20k3` | CFG-MP+, only the step count changed from the defaults | 20 | 3 | yes | 124 |
| `cfgmp_s10k1` | CFG-MP, only the round count changed from the defaults | 10 | 1 | no | 34 |
| `sweep_{cfg,cfgmpp}_w{3,5,7,9,12}` | guidance-scale sweep, 1 image per prompt | 30 / 10 | — / 3 | — / yes | 60 / 62 |

Large runs carry an `L_` prefix. The 10-step, K = 3 configuration is what the authors' released `demo_SD.py` runs; the paper's appendix reports SD3.5 ablations at 20 and 30 steps and recommends two rounds, and its main tables do not state the split. The port in `cfgmp_eval/pipeline.py` was checked against the authors' unmodified sampler: same seed, pixel-identical output for both variants (`scripts/check_port_against_authors.py`).

## Headline numbers

Paired differences against CFG at w = 4 (95% interval; **\*** = interval excludes zero). Full tables with all settings, per-task scores and the sweep are in [`tables/tables.md`](tables/tables.md); the JSON behind them is in `tables/{medium,large}/`.

| CFG-MP+ (demo defaults) − CFG | Medium | Large |
|---|---|---|
| GenEval | −0.0025 [−0.022, +0.017] | +0.007 [−0.009, +0.024] |
| HPSv2 | −0.73 [−0.78, −0.68]\* | −0.28 [−0.33, −0.24]\* |
| ImageReward | −0.046 [−0.078, −0.014]\* | −0.015 [−0.040, +0.010] |

Changing one knob at a time from the demo defaults: steps 10 → 20 (K = 3, Anderson fixed; 62 → 124 NFE) moves HPSv2 by +0.91\* on Medium and +0.47\* on Large; rounds 3 → 1 and Anderson vs plain each move it by less than 0.15 with signs that differ between models and metrics. CFG baselines: GenEval 0.683 [0.660, 0.707] on Medium, 0.714 [0.691, 0.736] on Large.

## Layout

| Path | Contents |
|---|---|
| `cfgmp_eval/pipeline.py` | CFG-MP / CFG-MP+ as `method` values on the harness's SD3.5 pipeline (port of the authors' sampler, MIT) |
| `cfgmp_eval/bootstrap.py` | paired prompt bootstrap, resampled within GenEval tasks, synchronized across settings |
| `scripts/generate.py` | image generation for one setting; seeds depend only on the prompt index; `--model` selects Medium or Large |
| `scripts/geneval_eval.py` | runs GenEval's own `evaluate_images.py` unchanged on mmdet 3.x; self-checks the mmcv deformable-attention kernel and falls back to mmcv's PyTorch implementation when the kernel is unusable (the case on H100) |
| `scripts/pref_scores.py` | HPSv2 and ImageReward on the same images (separate virtualenv) |
| `scripts/bootstrap_report.py` | scores, intervals and paired differences from `results.jsonl` / `pref_scores.jsonl` |
| `scripts/run_setting.sh`, `run_setting_sharded.sh`, `run_queue.sh`, `run_queue_large.sh` | generation + scoring for one setting, multi-GPU sharding, and the two queues that produced the results |
| `scripts/install_env.sh`, `install_pref.sh` | environments; they also fix two installer defects (mmdet 3.x needs the v3.0 re-keyed Mask2Former checkpoint; the hpsv2 wheel lacks its BPE vocab and imports `turtle`) |
| `scripts/with_keepalive.sh` | wrapper for a host that powers off when GPU utilisation is low |
| `configs/*.yaml` | one file per setting; each comment states what the code does at that value |
| `results/{medium,large}/<setting>/` | `results.jsonl` (GenEval per-image verdicts, unmodified evaluator output), `pref_scores.jsonl`, `summary.txt`, `run_info*.json` (planned NFE, dtype, model) |
| `results/medium/cfg/results_pytorch_msda.jsonl` | the same 2,212 images re-scored through the PyTorch deformable-attention path: identical verdicts to the CUDA path |
| `tables/` | bootstrap tables (JSON), `tables.md`, and `render_tables.py` that produced it |
| `figures/` | crop sheets: CFG vs CFG-MP+ at the demo settings, Medium and Large |
| `docs/` | review log (three claim audits and one external review), the claims sheet used for the audits (Korean) |
| `tests/` | unit tests for the NFE count, the sigma grid, Anderson convergence and the bootstrap |

Generated images are not included (44,240 files).

## Reproducing

1. Clone the harness and the CFG-MP repository next to this one (the harness has no license file, so it is not vendored):
   `git clone https://github.com/ThereWillComeSoftRains/RevisitingCFGMethods` (commit `51cc9ea`), `git clone https://github.com/LeonSuZhengYi/CFG-MP` (commit `ab848f9`).
2. `scripts/install_env.sh` (Python 3.10, torch 2.5.1+cu121, diffusers at the harness's pinned commit, mmdet 3.3.0 + mmcv 2.2.0, GenEval assets) and `scripts/install_pref.sh`. Accept the SD3.5 licences on Hugging Face and log in.
3. One setting: `scripts/run_setting.sh cfgmpp configs/cfgmpp.yaml` (add `--model stabilityai/stable-diffusion-3.5-large` for Large). Everything: `scripts/run_queue.sh` / `scripts/run_queue_large.sh`.
4. Tables: `python scripts/bootstrap_report.py --ref cfg,cfg0s cfg=results/medium/cfg/results.jsonl cfgmpp=results/medium/cfgmpp/results.jsonl` (use `--key hpsv2` or `image_reward` with the `pref_scores.jsonl` files).

Compute used: about 77 GPU-hours on one RTX A6000 (Medium, scoring included) and about 19 hours on three H100s (Large).

## Licenses

This repository is MIT-licensed (see `LICENSE`). `cfgmp_eval/pipeline.py` derives from the authors' MIT-licensed code; their notice is in `THIRD_PARTY_NOTICES.md`. Nothing from the unlicensed harness is redistributed.
