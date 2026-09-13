# 접근 14 — 접근 1 점수(cw_) 고도화 (Count-Weight Plus)

## 개요
9차 설정 모델의 분기 이득 77%가 접근 1의 클래스별 집약 점수 140개에서 나온다(3. docs/07 §6). 원본 열을 늘리는 피처는 트리가 안 쓰므로,
**새 정보를 같은 "클래스별 집약 점수" 형태로** 넣는다. 나머지(v4 피처, colsample 0.7, 3차 복원 배율, 쌍둥이 규칙)는 13차 구성 그대로 고정.
코드: `4. src/common/approach14_cw_plus/cw_plus.py`, 피처 종류 `--features v4p` (= v4 + cwp_*).

| 확장 | 토큰 | 뜻 | 열 수 |
|---|---|---|---|
| (i) 유형 분리 | `유전자:missense/lof/syn` (gene 단위), `유전자:변이` (missense만, variant 단위) | 같은 유전자라도 '어떻게' 바뀌었는지 구분한 NB 점수 | 28 + 28 |
| (ii) 위치 구간 | `유전자:b{위치//25}` (기능성 변이) | hotspot one-hot이 희소해 못 쓰인 정보를 25aa 구간으로 묶어 점수화 | 28 |
| (iii) 군집 상대 | 저변이 군집(SARC·PRAD·PCPG·THYM·OV), 상피암 군집(STES·LUSC·HNSC·LUAD·BLCA) 안 log-softmax | 군집 안에서만의 상대 순위 | 6 + 6 |

- 점수는 Bernoulli NB 로그우도비(토큰 수로 평균), float64 + 소수점 4자리(재현성), 학습 fold에서 2명 미만 토큰은 0.
- 학습 행은 내부 5-Fold OOF, **쌍둥이는 하나로만 센다**(접근 7의 붕괴 교훈). test는 fit 없이 transform.

## 버전
| 버전 | 파일 | 구성 | 정직 CV (배율 적용) | 저변이 군집 F1 | 상피암 군집 F1 | LB |
|---|---|---|---|---|---|---|
| v1 | (실행 중) | v4p + colsample 0.7 + 복원 배율 (+규칙) | | | | |
| (기준) 9차 | approach12_v4 | v4 + colsample 0.7 + 복원 배율 | 0.4847 | | | 0.4377 |
