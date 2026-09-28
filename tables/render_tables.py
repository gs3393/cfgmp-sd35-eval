"""bootstrap_report.py가 저장한 JSON 표(medium/, large/)를 보고서 부록용 Markdown 표로 만든다.

    python -X utf8 reports/tasks/250790696/data/render_tables.py > reports/tasks/250790696/data/tables.md
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = {
    "cfg": "CFG · N=30 · 60 NFE",
    "cfg0s": "CFG-Zero* · N=30 · 60 NFE",
    "apg": "APG · N=30 · w=7 · 60 NFE",
    "cfgmp": "CFG-MP(Picard) · N=10 · K=3 · 62 NFE",
    "cfgmpp": "CFG-MP+(AA) · N=10 · K=3 · 62 NFE (demo 기본값)",
    "s20k1": "Picard · N=20 · K=1 · 68 NFE (`cfgmpp_s20k1`, AA 비활성)",
    "s20k3": "CFG-MP+(AA) · N=20 · K=3 · 124 NFE",
    "s10k1": "CFG-MP(Picard) · N=10 · K=1 · 34 NFE",
}
ORDER = ["cfg", "cfg0s", "apg", "cfgmp", "cfgmpp", "s20k1", "s20k3", "s10k1"]
TASKS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]


def load(model, name):
    with open(os.path.join(HERE, model, name), encoding="utf-8") as f:
        return json.load(f)


def num(x, nd):
    return f"{x:.{nd}f}"


def diff(t, n, ref, nd):
    v = t[n].get(f"diff_vs_{ref}")
    if not v:
        return "—"
    return f"{v['diff']:+.{nd}f} [{v['ci'][0]:+.{nd}f}, {v['ci'][1]:+.{nd}f}]" + ("★" if v["resolved"] else "")


def main_table(model, pre):
    g, h, r = (load(model, f"{pre}_main_{k}.json") for k in ("geneval", "hpsv2", "image_reward"))
    out = ["| 설정 | GenEval [95%] | Δ vs CFG | HPSv2 | Δ vs CFG | ImageReward | Δ vs CFG |", "|---|---|---|---|---|---|---|"]
    for n in ORDER:
        if n not in g:
            continue
        out.append(f"| {SETTINGS[n]} | {num(g[n]['score'],4)} [{num(g[n]['ci'][0],3)}, {num(g[n]['ci'][1],3)}] | {diff(g,n,'cfg',4)} | "
                   f"{num(h[n]['score'],2)} | {diff(h,n,'cfg',2)} | {num(r[n]['score'],3)} | {diff(r,n,'cfg',3)} |")
    return "\n".join(out)


def task_table(model, pre):
    g = load(model, f"{pre}_main_geneval.json")
    names = [n for n in ORDER if n in g]
    short = {"cfg": "CFG", "cfg0s": "CFG-0*", "apg": "APG", "cfgmp": "MP 10/3", "cfgmpp": "MP+ 10/3", "s20k1": "Picard 20/1", "s20k3": "MP+ 20/3", "s10k1": "MP 10/1"}
    out = ["| task | " + " | ".join(short[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for t in TASKS:
        out.append(f"| {t} | " + " | ".join(num(g[n]["per_task"][t], 3) for n in names) + " |")
    out.append("| **Overall** | " + " | ".join(f"**{num(g[n]['score'],3)}**" for n in names) + " |")
    return "\n".join(out)


def sweep_table(model, pre):
    out = ["| w | CFG GenEval | CFG-MP+ GenEval | Δ GenEval | CFG HPSv2 | Δ HPSv2 | CFG IR | Δ IR |", "|---|---|---|---|---|---|---|---|"]
    for w in (3, 5, 7, 9, 12):
        g, h, r = (load(model, f"{pre}_sweep_w{w}_{k}.json") for k in ("geneval", "hpsv2", "ir"))
        out.append(f"| {w} | {num(g['cfg']['score'],3)} | {num(g['cfgmpp']['score'],3)} | {diff(g,'cfgmpp','cfg',3)} | "
                   f"{num(h['cfg']['score'],2)} | {diff(h,'cfgmpp','cfg',2)} | {num(r['cfg']['score'],3)} | {diff(r,'cfgmpp','cfg',3)} |")
    return "\n".join(out)


def isolation_table():
    pairs = [("s20k3", "cfgmpp", "sampling step N: 10 → 20 (K=3, AA 고정; NFE 62 → 124)"),
             ("s10k1", "cfgmp", "projection 반복 K: 3 → 1 (N=10, Picard 고정; NFE 62 → 34)"),
             ("cfgmpp", "cfgmp", "Anderson vs Picard (N=10, K=3 고정; NFE 62)"),
             ("s20k1", "cfg", "Picard N=20 K=1 (68 NFE) vs CFG N=30 (60 NFE): 예산 근접")]
    out = ["| 달라진 요인 | 모델 | Δ GenEval | Δ HPSv2 | Δ ImageReward |", "|---|---|---|---|---|"]
    for a, b, what in pairs:
        for model, pre, label in (("medium", "M", "Medium"), ("large", "L", "Large")):
            g, h, r = (load(model, f"{pre}_main_{k}.json") for k in ("geneval", "hpsv2", "image_reward"))
            out.append(f"| {what if label == 'Medium' else ''} | {label} | {diff(g,a,b,4)} | {diff(h,a,b,2)} | {diff(r,a,b,3)} |")
    return "\n".join(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print('표기: 점수는 세 지표 모두 6개 task의 macro 평균. 대괄호는 쌍별 짝지은 bootstrap 95% 백분위 구간(10,000회, task 안에서 prompt 재추출, 설정 간 동기화), ★ 는 구간이 0을 제외함. 재평가 논문이 쓰는 방법들에 걸쳐 pooled한 95백분위 공유 margin과는 판정 기준이 다르다. 원본: `data/medium/*.json`, `data/large/*.json`.\n')
    print("### A.1 Medium 주 설정 (553 prompt × 4장, w=4 외 표기)\n"); print(main_table("medium", "M")); print()
    print("### A.2 Large 주 설정 (553 prompt × 4장, w=4)\n"); print(main_table("large", "L")); print()
    print("### A.3 요인 분리: 한 요인만 다른 쌍의 짝지은 차이\n"); print(isolation_table()); print()
    print("### A.4 task별 GenEval — Medium\n"); print(task_table("medium", "M")); print()
    print("### A.5 task별 GenEval — Large\n"); print(task_table("large", "L")); print()
    print("### A.6 guidance scale sweep — Medium (553 prompt × 1장, CFG N=30 vs CFG-MP+ N=10/K=3)\n"); print(sweep_table("medium", "M")); print()
    print("### A.7 guidance scale sweep — Large (553 prompt × 1장)\n"); print(sweep_table("large", "L"))


if __name__ == "__main__":
    main()
