# 주장 시트 — 카드 250790696 (CFG-MP vs SD3.5 재평가)

hook이 매 턴 컨텍스트에 넣는 파일. 원문·코드에서 그대로 옮긴 문장과 확인된 부재만 둔다. "논문에 없다"고 쓰려면 여기에 그 부재가 적혀 있거나 research-archive에서 다시 찾은 뒤에 쓴다. 수치의 원본은 결과 파일이다.

## CFG-MP 원문 (arXiv 2601.21892, ICML 2026)
- 모델: "We leverage SD3.5 (Esser et al., 2024) and Flux-dev (BlackForestLabs et al., 2025) as our backbone text-to-image generation models." Medium/Large, 해상도, checkpoint 이름은 본문·부록·저장소 어디에도 없다(2026-09-21 전문 검색).
- 설정: "For our proposed CFG-MP+, we consistently employ the AA(1,1) configuration." 주 표(Table 2·3)의 step/반복 배분, time_threshold, H/G 전환점은 본문에 없다. **부록에는 있다**: Table 6 "Ablation study of CFG-MP+ with varying λ on HPSv2 and IR metrics"는 SD3.5를 Steps 20·30에서 돌렸고(λ=0.5: HPSv2 31.18/31.21, IR 1.12/1.12), FPI ablation은 "we recommend setting FPI = 2 in practice", Table 8은 SD3.5에서 K∈{3,4,5}. 이 카드의 "저자 기본값" 10 step/K=3/window 1/threshold 0.6은 저장소 `CFG-MP_SD/demo_SD.py`와 README 표의 값이다. `utils_SD.py`의 `__call__` 자체 기본값은 28 step, w=4.5, window 2다.
- 하드웨어: "All experiments are conducted on a 4-A100(40GB) GPU server." Table 7(SD3.5, NFE 60): CFG 15.43 s/장, CFG-MP+ 15.04 s/장.
- Table 3 (GenEval, "SD3.5 (cfg=4, NFE=60)", %): CFG 58.36 / CFG-0* 59.24 / R-CFG++ 59.57 / CFG-MP 60.81 / CFG-MP+ 61.41. Position: CFG 13.00. 오차·구간 없음.
- Table 2 (SD3.5, Pick-a-Pic, HPS/IR): w=3 NFE60 CFG 29.69/0.99, CFG-MP+ 30.78/1.11. w=4 NFE60 CFG 30.18/1.04, CFG-MP+ 30.94/1.16. 주장 A: CFG-MP+가 CFG-0*·R-CFG++보다 HPSv2·IR 높음.
- 주장 B(scale 민감도 완화): 초록·서론은 "reduces sensitivity to the guidance scale". 정량 근거는 DiT-XL/ImageNet의 FID·IS(ω∈{1.5,2.0,2.5}, 부록 ratio·GIS(ω) 분석)이고, SD3.5에서는 w=3, 4 두 값만 보고했다. SD3.5 scale sweep은 원문에 없다.

## 재평가 논문 (arXiv 2608.16786, 2026-08)
- 프로토콜: "Methods share prompts, seeds, scheduler, resolution, and sampling budget within each model: 25 steps for SD3.5 and 30 for FLUX; inference used V100 32GB GPUs." "We use 10,000 paired prompt-bootstrap replicates with synchronized sampling for every method". "Differences inside it are treated as unresolved rather than as evidence of equivalence."
- 오차 판정: "For each benchmark, centered method–CFG errors are pooled over guided alternatives; their 95th percentile is the shared bootstrap margin." → 공유 margin 방식. 이 카드의 bootstrap은 쌍별 95% 백분위 구간이라 판정 기준이 다르다.
- SD3.5 GenEval에서 오차 밖 이득은 SAG(0.715, +0.060) 하나: "On SD3.5, SAG produces the largest GenEval score and improves reasoning beyond the margin." 날짜: CFG-MP v2 2026-05-13, 재평가 v1 2026-08-17.
- SD3.5 Medium GenEval: CFG 0.655, APG 0.671(오차 안), 95% Δmargin ±0.027(차이의 공유 margin이지 CFG 점수의 구간이 아님). Table 3 tuned 값: CFG-Zero* w=3.5, zero-init steps=0. fp16·prompt당 1장은 논문이 아니라 harness 저장소 설정(`pipelines/sd35/*.py` float16, `configs/campaigns/main.yaml` samples_per_prompt 1).
- **표 산술 관찰**: Table 5의 SD3.5 Medium 열은 task마다 분모가 30·31의 **배수**일 때만 모든 셀이 재현된다(CFG two 0.833=25/30, pos 0.226=7/31=14/62; 어느 배수인지 미정). 전체 553 prompt 개수(80/99/80/94/100/100)로는 prompt당 11·12장일 때만 맞음(harness는 1장). FLUX 열은 553 분모로 재현(0.988=79/80, 0.889=88/99, 0.560=56/100). 본문 "We use the full prompt sets on selected benchmarks". **표본 수는 확정 불가** — "약 184 prompt" 추정과 CFG-Zero* 차이 원인 연결은 2026-09-28 외부 검토로 철회. 저자 확인 없음.
- 결론 원문: "Only a small number of their nominal gains exceed the shared margins, whereas resolved decreases occur across several methods and metrics." SD3.5: "CFG++ and CFG-Zero* decrease reasoning, CFG++ also decreases GenEval, and APG and TCFG have resolved losses on reasoning. … PAG has resolved losses on GenEval, DPG-Bench, text rendering, and reasoning."
- → **오차 밖 악화는 이 논문에도 여러 건 있다.** 이 논문이 다루지 않은 것: CFG-MP/MP+(8개 변형에 없음), HPSv2·ImageReward, 정량 scale sweep(부록 C에 정성 sweep 그림만 있음).

## 코드 사실 (gs3393/cfgmp-sd35-eval; 저자 utils_SD.py와 픽셀 동일 확인)
- demo 기본값(`demo_SD.py`; w만 Table 3의 4로 치환): 10 step, max_aa_iter=3, aa_window_size=1, time_threshold=0.6 → 62 NFE. t_curr ≥ 0.95면 H, 아니면 G 연산자.
- Anderson은 이력이 2개 이상일 때만 작동한다: `m = len(f_history) - 1 < 1`이면 `z + β·f` = Picard. K=3이면 2·3번째 반복만 AA. **K=1이면 AA 분기가 작동하지 않아 CFG-MP(Picard) 1회와 같다. `cfgmpp_s20k1`·`L_cfgmpp_s20k1`은 CFG-MP+가 아니라 Picard 20 step × 1회다.**
- NFE = 2N + 2K·(t_curr > 0.6인 step 수). 10/3→62, 20/1→68, 20/3→124, 10/1→34, 30/1→100. **30/1이 원문 NFE 100과 맞아도 원문 설정일 수 없다**: K=1이면 MP+≡MP인데 원문은 NFE 100에서 둘을 다르게 보고(Table 7 HPSv2 30.78 vs 31.15).
- 설정 비교 규칙: 두 설정 사이에 달라진 요인(step 수 N, 반복 수 K, AA/Picard, w, NFE)을 전부 나열한 뒤에 원인을 말한다. 10/3/AA vs 20/1은 세 요인이 동시에 다르다.

## 현재 결과 요약 (2026-09-28 완료. 원본: outputs/<설정>/results.jsonl·pref_scores.jsonl, 표: reports/tasks/250790696/data/{medium,large}/*.json)
표기: 짝지은 bootstrap 95% 구간이 0을 제외하면 *. 주 설정은 553 prompt × 4장, w=4, CFG는 30 step(60 NFE). demo 기본 = 10 step/K=3/AA(62 NFE).
- **주장 A(GenEval)**: 저자기본 CFG-MP+ vs CFG — Medium −0.0025 [−0.022, +0.017], Large +0.007 [−0.009, +0.024]. 원문의 +0.03은 두 구간 모두 밖. CFG-MP(Picard)도 Medium −0.013, Large +0.011로 안.
- **선호 점수**: 저자기본은 HPSv2에서 두 모델 모두 오차 밖으로 낮음(CFG-MP+ Medium −0.73*, Large −0.28*; CFG-MP −0.84*/−0.37*). IR은 Medium −0.046*, Large 안.
- **요인 분리(한 요인만 다름)**: N 10→20(K=3, AA 고정, NFE 62→124): HPSv2 +0.91*(M)/+0.47*(L), IR +0.16*/+0.054*, GE +0.034*/+0.007. K 3→1(N=10, Picard 고정): HPSv2 +0.15*(M)/−0.01(L), IR −0.01/−0.05*. AA vs Picard(10/3): HPSv2 +0.11*/+0.08*, IR +0.023*/−0.013*. → 시험한 세 변경 중 step 수 효과가 가장 큼, 나머지 둘은 ≤0.15.
- **예산 근접 비교**: s20k1(Picard 20×1, 68 NFE) vs CFG(60 NFE): Medium GE +0.002, HPSv2 +0.02, IR +0.042*; Large GE +0.007, HPSv2 +0.053*, IR +0.007. GE 미해결(동등의 증거 아님), 선호 지표 일부(M IR, L HPSv2)에서 작은 양의 차이. 비용 68 vs 60. s20k3(124 NFE)이 CFG를 이기는 것(M: GE +0.032*, HPSv2 +0.18*, IR +0.11*; L: HPSv2 +0.18*, IR +0.039*)은 예산 2배이며 CFG 62 step 대조군은 없음.
- **주장 B(scale sweep, 553×1, 저자기본 10 step)**: 두 모델 모두 반대. Medium w=9 GE −0.095*, w=12 −0.215*, HPSv2 −2.21*; Large w=7 GE −0.051*, w=12 −0.126*, HPSv2 w=3~12 전부 밖(−0.32→−1.13). 20 step에서의 sweep은 하지 않았음(한계).
- **CFG-MP+ − CFG-Zero***: Medium GE +0.022*, IR +0.074*, HPSv2 −0.50*(CFG-Zero*가 CFG 아래인 결과); Large GE +0.007, HPSv2 −0.19*, IR −0.010.
- **baseline**: APG(Medium) 세 지표 모두 안(재평가 논문과 일치). CFG-Zero*(Medium, harness w=3.5) 세 지표 모두 밖으로 낮음(GE −0.024*); Large(w=4) GE 안, HPSv2 −0.09*.
- **CFG 기준선**: Medium GE 0.683 [0.660, 0.707](재평가 0.655와 다름 — 그 논문은 CFG 점수의 구간을 주지 않음; 원문 58.36과 10pt 차이), Large 0.714.
- **채점 경로 대조**: Medium CFG 2,212장을 A6000에서 CUDA 커널/PyTorch MSDA로 채점 → 판정 차이 0장(박스 수치는 1,815장 차이). H100(PyTorch) 결과와 A6000 결과는 판정 수준에서 비교 가능.
- 미실행: CFG 62 step(124 NFE 대조군), 20 step에서의 scale sweep, DPG·OneIG.

## 정정 이력
- 2026-09-28 외부 검토 4건 반영: 재평가 표본 수 추정 철회, 미해결을 동률로 쓰지 않음, 질감→GenEval 서술 완화(GenEval은 개수·색·위치·속성도 잰다), 설치 결함 영향은 GenEval 경로만(HPSv2·IR 무관).
- 2026-09-28 대조 에이전트 지적 반영: 원문 부록의 20·30 step·FPI=2 권고를 놓쳤음("원문이 배분을 밝히지 않았다"는 주 표에 한정해야 함), 재평가 margin은 공유 margin, 재평가 SD3.5 표는 소표본 추정, IR 범위 +0.01~+0.13, HPSv2·IR도 macro 평균.
- 2026-09-24: "악화 자체가 재평가 논문에 없던 발견" → 틀림(위 결론 원문). "cfgmpp_s20k1 = CFG-MP+ 20 step" → 틀림(Picard). "10/3 vs 20/1 차이는 step 배분 때문" → 세 요인 미분리.
