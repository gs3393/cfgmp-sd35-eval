# cfgmp-sd35-eval

CFG-MP / CFG-MP+ ([arXiv:2601.21892](https://arxiv.org/abs/2601.21892))를 SD3.5-Medium 재평가 harness
([RevisitingCFGMethods](https://github.com/ThereWillComeSoftRains/RevisitingCFGMethods), arXiv:2608.16786)와
같은 잣대로 재는 개인 검증용 코드다. 작업 카드: <https://github.com/users/gs3393/projects/1?pane=issue&itemId=250790696>

## 구성

| 경로 | 내용 |
|---|---|
| `cfgmp_eval/pipeline.py` | harness의 `StableDiffusion3MethodsPipeline`을 상속해 `method: cfgmp`(Picard) / `cfgmpp`(Anderson)를 추가. 저자 코드(`CFG-MP_SD/utils_SD.py`, MIT)의 sampling loop를 옮긴 것 |
| `cfgmp_eval/bootstrap.py` | task 안에서 prompt를 재추출하는 짝지은 bootstrap (재평가 논문의 절차) |
| `scripts/generate.py` | 설정 하나의 GenEval 이미지를 생성. 모든 설정이 `seed + prompt index`의 같은 noise에서 출발 |
| `scripts/geneval_eval.py` | mmdet 3.x 출력을 2.x 형식으로 바꿔 GenEval 원본 `evaluate_images.py`를 그대로 실행 |
| `scripts/bootstrap_report.py` | 설정별 점수·신뢰구간과 기준 설정 대비 짝지은 차이 |
| `scripts/run_setting.sh` | 생성 → GenEval 채점 → 요약 |
| `configs/` | 설정별 YAML. baseline은 harness의 tuned 값, CFG-MP는 저자 demo 기본값 |

harness 저장소는 라이선스가 없어 이 저장소에 복사하지 않는다. 옆 디렉터리에 clone해 `HARNESS_ROOT`로 가져다 쓴다.

## 비교의 기준

- **NFE를 맞춘다.** CFG-MP는 projection 반복에 forward를 더 쓴다. 저자 기본값(10 step, 반복 3회, t > 0.6에서만 projection)은
  이미지당 62 NFE다. baseline은 30 step = 60 NFE로 둔다. harness 원래 값(25 step)은 `cfg_harness25.yaml`로 따로 둔다.
- dtype은 모든 설정에서 bfloat16, 해상도 1024², prompt당 4장.

## 실행 (원격 `gpu-host`)

```bash
WS=~/data/code/cfgmp-sd35-eval-ws
scripts/run_setting.sh cfg configs/cfg.yaml
scripts/run_setting.sh cfgmpp configs/cfgmpp.yaml
python scripts/bootstrap_report.py --ref cfg cfg0s \
    cfg=$WS/outputs/cfg/results.jsonl cfgmpp=$WS/outputs/cfgmpp/results.jsonl
```

테스트: `python -m pytest -q tests` (GPU 불필요).
