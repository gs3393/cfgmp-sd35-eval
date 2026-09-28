# Review log

The report and the note went through three claim audits by an agent that had not seen the work, plus one external review. Each audit compared every number against the JSON tables, every statement about the two papers against the papers' text, and every statement about the code against the source. Findings and what changed are listed below; the claims sheet the audits were checked against is `claims-sheet.ko.md` (Korean).

## Audit 1 — report draft, 2026-09-28

21 findings (3 errors, 4 unsupported, 6 overclaims, 8 minor). All applied. The ones that changed the framing:

- The CFG-MP paper's appendix runs SD3.5 at 20 and 30 steps (its λ ablation) and recommends two projection rounds; the main tables do not state the split. "The paper does not state the step split" had to be limited to the main tables, and the 10-step, K = 3 configuration had to be named as the released demo's, not the paper's.
- The re-evaluation preprint decides significance with a shared margin (the 95th percentile of centered method–CFG errors pooled over methods); this work uses per-pair 95% intervals. The two rules are related but not the same, and the text now says so.
- HPSv2 and ImageReward scores are macro averages over the six GenEval tasks, as GenEval is; an earlier legend called them weighted averages.
- The ImageReward range the paper reports over CFG is +0.01 to +0.13, not +0.07 to +0.12.

## Audit 2 — revised report, 2026-09-28

11 findings (2 overclaims, 9 minor). All applied.

- "20 steps with K = 3 is close to the paper's appendix condition" was narrowed: the appendix's step table does not state K, its K table does not state the steps, and 124 NFE is not a budget the paper reports.
- The clue "30 steps × 1 round = exactly 100 NFE" was withdrawn: with one round the Anderson variant is identical to the plain one, but the paper reports the two differently at 100 NFE.
- Config comments that called the demo values "the paper's" were corrected.

## Audit 3 — blog note draft, 2026-09-28

17 findings (3 errors, 2 overclaims, 12 minor). All applied.

- The preprint (August 2026) is three months after the paper's revised arXiv version (May 2026), not before it.
- The preprint's one GenEval gain beyond its margin on SD3.5 is SAG's, not APG's.
- The "recommend FPI = 2" sentence belongs to the text-to-image ablation measured in HPSv2 and ImageReward, not to the DiT ablation.
- Against CFG-Zero* on Medium, CFG-MP+ is above it on GenEval and ImageReward (and below on HPSv2) because CFG-Zero* at the harness's tuned w = 3.5 lands below CFG itself; this comparison was missing from the report and was added.

## External review, 2026-09-28

Four items, all accepted:

1. The per-task GenEval values in the preprint's SD3.5 table reproduce with any multiple of 30 or 31 as the denominator (7/31 = 14/62), and with the full 553-prompt set at 11 or 12 images per prompt, so a sample size cannot be inferred from the table. The "about 184 prompts" estimate and its use as a candidate explanation for the CFG-Zero* discrepancy were withdrawn; the note dropped the section.
2. Unresolved differences were being summarized as "a tie"; the protocol says an interval containing zero is not evidence of equivalence. Reworded throughout.
3. "GenEval only asks whether the right objects are there" was wrong (it also scores counts, colors, positions and attribute binding), and the texture observation was not quantified against the HPSv2 gap. Narrowed.
4. The two installer defects affect the GenEval detector path only; HPSv2 and ImageReward come from separate models. The observed values (no detections on the demo photo; 4 of 16 correct on the H100 smoke test) replaced "every number would have been zero".

"Step count is the dominant factor" was narrowed to "the change with the largest effect among the three tried".

The review text (Korean, verbatim):

```
PR #24와 블로그 초안 PR #26을 검토했습니다. 핵심 결과는 유지할 수 있지만, 아래 4건을 보완한 뒤 병합하는 것이 좋겠습니다. 현재 검토에서 새 실험이 필요한 문제는 발견하지 못했습니다.

1. [P2] 표의 분모만으로 "약 184개 prompt"를 추정한 논리가 성립하지 않습니다. 보고서는 가능한 분모가 30·31뿐이라고 하지만, 원문 표 전체를 대조하면 60·62 등도 가능합니다. 7/31 = 14/62이므로 368개 prompt로도 같은 표를 설명할 수 있습니다. "184개 위에서 계산된 margin"이라는 추론과 이를 CFG-Zero* 결과 차이의 원인 후보로 연결한 부분은 내려야 합니다. "작은 분모의 비율들과 일치하지만 실제 표본 수는 확인되지 않았다"까지는 가능합니다. 블로그에서는 이 절을 빼는 편을 권합니다.

2. [P2] 미해결 결과를 결론에서 다시 "동률"로 바꾸고 있습니다. 프로토콜에서는 구간이 0을 포함해도 동등의 증거가 아니라고 정확히 설명합니다. 그런데 결론은 "예산을 맞춘 … CFG와 동률", 초안은 "A tie with a slight edge"라고 합니다. 해당 비교는 68 대 60 NFE로 비용도 다릅니다. "계산량이 근접한 비교에서 GenEval 차이는 미해결이고, 일부 선호 지표에서 작은 양의 차이를 관찰했다"로 통일해야 합니다.

3. [P2] 질감 결함이 지표 차이를 설명한다고 단정하는 문장이 남았습니다. 초안은 "물체는 괜찮으므로, 물체 존재만 확인하는 GenEval은 알아채지 못한다"고 설명합니다. GenEval은 개수·색·위치·속성 결합도 평가합니다. 또한 네 prompt의 그림으로 질감 결함이 HPSv2 격차의 원인인지 확인한 것은 아닙니다. "질감 결함을 관찰했으며, GenEval은 이를 직접 평가하도록 설계된 지표가 아니다"로 좁히면 됩니다. 뒤의 한계 절에도 정량 연결을 하지 않았다고 이미 적혀 있습니다.

4. [P2] 설치 결함이 "위의 모든 수치를 0으로 만들었을 것"이라는 설명은 틀립니다. 두 결함은 GenEval 검출기 경로의 문제입니다. 별도 채점기인 HPSv2·ImageReward까지 0으로 만들지는 않습니다. H100의 실제 관찰도 16장 중 4장 정답이므로 "아무것도 검출하지 못함"과 다릅니다. "GenEval 평가를 심각하게 손상시켰다"로 고치고 실제 관찰값을 쓰는 것이 정확합니다.

앞서 문제였던 Picard/Anderson 구분, demo 설정과 논문 설정의 구분, bootstrap 판정 방식 차이, 124 NFE 비교의 한계는 잘 반영됐습니다. "step 수가 지배적"이라는 표현도 "이번에 시험한 변경 중 HPSv2 효과가 가장 컸다"로 한정하면 더 정확합니다.

첨부 JSON 36개의 평균·차이·구간 판정을 확인했고, 렌더러 출력은 첨부 표와 보고서 부록에 모두 일치했습니다. 두 그림도 확인했습니다. 원격의 이미지별 원본과 실행 로그까지 재검증한 것은 아닙니다.
```
