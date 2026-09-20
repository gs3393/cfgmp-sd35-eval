"""Paired prompt bootstrap over per-image scores.

Protocol follows "Revisiting CFG Methods" (arXiv:2608.16786), Sec. on uncertainty: prompts are resampled
with replacement *within each task*, the same resampled prompts are used for every setting (synchronized),
and the headline score is the macro average over tasks. A prompt's score is the mean over its images.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_per_prompt(results_jsonl: str, key: str = "correct") -> Dict[int, Tuple[str, float]]:
    """{prompt index: (task tag, mean score over that prompt's images)} from a GenEval-style results file."""
    per_prompt = defaultdict(list)
    tags = {}
    for line in open(results_jsonl):
        row = json.loads(line)
        idx = int(Path(row["filename"]).parent.parent.name)
        per_prompt[idx].append(float(row[key]))
        tags[idx] = row["tag"]
    return {idx: (tags[idx], float(np.mean(vals))) for idx, vals in per_prompt.items()}


def align(settings: Dict[str, Dict[int, Tuple[str, float]]]):
    """Keep prompts present in every setting. Returns (tags array, {setting: scores array}, n_dropped)."""
    common = sorted(set.intersection(*(set(v) for v in settings.values())))
    n_dropped = max(len(v) for v in settings.values()) - len(common)
    first = next(iter(settings.values()))
    tags = np.array([first[i][0] for i in common])
    scores = {name: np.array([v[i][1] for i in common]) for name, v in settings.items()}
    return tags, scores, n_dropped


def macro_score(tags: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean([scores[tags == t].mean() for t in np.unique(tags)]))


def paired_bootstrap(tags: np.ndarray, scores: Dict[str, np.ndarray], n_boot: int = 10000, seed: int = 0):
    """Returns {setting: array[n_boot]} of macro scores, all computed on the same resampled prompts."""
    rng = np.random.default_rng(seed)
    task_idx = [np.flatnonzero(tags == t) for t in np.unique(tags)]
    out = {name: np.zeros(n_boot) for name in scores}
    for members in task_idx:
        draw = members[rng.integers(0, len(members), size=(n_boot, len(members)))]  # [n_boot, n_task]
        for name, s in scores.items():
            out[name] += s[draw].mean(axis=1)
    for name in out:
        out[name] /= len(task_idx)
    return out


def summarize(tags, scores, references: List[str], n_boot: int = 10000, seed: int = 0, level: float = 0.95):
    boots = paired_bootstrap(tags, scores, n_boot, seed)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    table = {}
    for name, s in scores.items():
        row = {
            "score": macro_score(tags, s),
            "ci": [float(x) for x in np.percentile(boots[name], [lo, hi])],
            "per_task": {t: float(s[tags == t].mean()) for t in np.unique(tags)},
        }
        for ref in references:
            if ref == name or ref not in scores:
                continue
            diff = boots[name] - boots[ref]
            d_lo, d_hi = (float(x) for x in np.percentile(diff, [lo, hi]))
            row[f"diff_vs_{ref}"] = {
                "diff": macro_score(tags, s) - macro_score(tags, scores[ref]),
                "ci": [d_lo, d_hi],
                "resolved": bool(d_lo > 0 or d_hi < 0),  # CI excludes 0
            }
        table[name] = row
    return table
