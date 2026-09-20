"""Paired-bootstrap table over GenEval-style results files.

    python scripts/bootstrap_report.py --ref cfg cfg0s cfg=out/cfg/results.jsonl cfgmpp=out/cfgmpp/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfgmp_eval.bootstrap import align, load_per_prompt, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", nargs="+", metavar="NAME=RESULTS_JSONL")
    ap.add_argument("--ref", nargs="*", default=["cfg"], help="settings to take paired differences against")
    ap.add_argument("--key", default="correct", help="per-image score field")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the full table here")
    args = ap.parse_args()

    settings = {}
    for item in args.settings:
        name, path = item.split("=", 1)
        settings[name] = load_per_prompt(path, args.key)
    tags, scores, n_dropped = align(settings)
    if n_dropped:
        print(f"warning: {n_dropped} prompts missing from at least one setting were dropped")
    table = summarize(tags, scores, args.ref, args.n_boot, args.seed)

    print(f"{len(tags)} prompts, {args.n_boot} replicates, key={args.key}")
    for name, row in table.items():
        line = f"{name:>14}  {row['score']:.4f}  [{row['ci'][0]:.4f}, {row['ci'][1]:.4f}]"
        for ref in args.ref:
            d = row.get(f"diff_vs_{ref}")
            if d:
                mark = "resolved" if d["resolved"] else "inside margin"
                line += f"  | vs {ref}: {d['diff']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] {mark}"
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
