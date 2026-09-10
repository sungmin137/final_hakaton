# 접근 10 — 혼동 암종 쌍 재판기

기존 V4가 만든 상위 두 암종이 미리 정한 혼동 쌍이고 확률 차이가 작을 때만, 해당 두 암종의 학습 환자만으로 학습한 Logistic Regression이 원본 유전자 변이 유무를 다시 판정한다.

- 1차 모델: V4 XGBoost (기존 공식 파라미터)
- 2차 모델: 원본 유전자별 WT/변이 여부, pairwise Logistic Regression
- 고정 후보 쌍: BRCA–OV, BRCA–PRAD, HNSC–STES, LIHC–STES, LUAD–STES, LUSC–STES, OV–PAAD
- 제외: KIRC–KIPAN, LGG–GBMLGG. 완전 동일한 유전체 프로필에 서로 다른 라벨이 붙은 쌍둥이 충돌이라 일반 재판 학습 대상으로 삼지 않는다.
- 검증: `StratifiedGroupKFold`로 쌍둥이를 같은 fold에 묶는다. test.csv는 읽지 않는다.

실행:

```bash
PYTHONPATH="4. src/common" .venv/bin/python "4. src/common/approach10_pair_referee_cv.py"
```

결과는 `6. experiments/2026-09-10_approach10_pair_referee/result.json`에 저장한다.

## 결과 (2026-09-10)

| 방식 | 쌍둥이 그룹 OOF Macro F1 | Accuracy |
|---|---:|---:|
| V4 기본 | 0.4683 | 0.4733 |
| 재판기 (확률 차이 0.30 이하) | **0.4749** | **0.4796** |

- V4의 최종 답을 바꾼 환자: 408명
- 틀린 답에서 맞는 답으로 바뀜: 115명
- 맞는 답에서 틀린 답으로 바뀜: 76명
- 각 fold에서 나머지 4개 fold만 보고 문턱값을 고르는 교차선택도 모두 `0.30`을 선택했고, 같은 0.4749가 나왔다.
- 위 CV 단계에서는 test.csv를 읽지 않았다. 실제 리더보드 이득 여부는 최종 제출 파일을 업로드한 뒤에만 알 수 있다.

## 최종 추론 구성

`approach10_v1_20260910.py`는 최종 추론에서만 test를 읽고 다음 순서로 답을 만든다.

1. 전체 train으로 V4를 학습한다.
2. 기존 V4의 train OOF만으로 클래스 배율을 구해 확률에 적용한다.
3. **보정 전** V4의 상위 두 답이 접근10의 혼동 쌍이고 차이가 0.30 이하인 test 행만 재판기로 바꾼다. 이 조건은 CV에서 검증한 조건 그대로다.
4. TwinRule 적용 전/후 파일을 각각 저장한다.

클래스 배율과 재판기를 동시에 적용한 CV는 별도로 측정하지 않았으므로, 이 제출의 예상 LB는 과거 V4+배율+TwinRule 최고점 0.4369를 기준으로 약 0.43~0.44 범위로만 본다. 실제 점수는 업로드 뒤에만 알 수 있다.

### 생성 결과 (2026-09-10)

- 기본본: `5. submissions/approach10_pair_referee_v1_20260910.csv`
- 권장본: `5. submissions/approach10_pair_referee_v1_20260910_twin_rule.csv`
- 두 파일 모두 2,546행, `ID` 순서 일치, 26개 라벨, 결측 0개로 검증했다.
- 클래스 배율 후 재판기가 바꾼 답: 119행. TwinRule이 추가로 바꾼 답: 7행.
- 업로드는 수행하지 않았다.
