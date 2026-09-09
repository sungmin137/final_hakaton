# 데이터 분석 전 알아야 할 도메인 지식 (2026-09-09)

> 모든 수치는 train.csv 기준. test.csv는 열지 않음.

## 1. 데이터의 정체: TCGA 체세포 변이 (somatic mutation) 프로파일
- 라벨 26개는 **TCGA(The Cancer Genome Atlas) 암종 코드**. 각 행은 환자 1명의 종양 시퀀싱 결과.
- 컬럼 4,384개는 유전자 이름(HGNC symbol). 값은 그 유전자에서 발견된 **단백질 수준 변이**.
- `WT` = wild type = 이 유전자에 변이 없음. 99.2%가 WT이므로 극도로 희소한 데이터.

### TCGA 코드 뜻 (26개)
| 코드 | 암종 | 코드 | 암종 |
|---|---|---|---|
| ACC | 부신피질암 | LUAD | 폐선암 |
| BLCA | 방광암 | LUSC | 폐편평상피암 |
| BRCA | 유방암 | OV | 난소암 |
| CESC | 자궁경부암 | PAAD | 췌장암 |
| COAD | 대장암 | PCPG | 갈색세포종/부신경절종 |
| DLBC | 미만성 거대 B세포 림프종 | PRAD | 전립선암 |
| GBMLGG | 교모세포종+저등급 신경교종 (합본) | SARC | 육종 |
| HNSC | 두경부 편평상피암 | SKCM | 피부 흑색종 |
| KIPAN | 신장암 전체 (KIRC+KIRP+KICH 합본) | STES | 위암+식도암 (합본) |
| KIRC | 신장 투명세포암 | TGCT | 고환 생식세포종양 |
| LAML | 급성 골수성 백혈병 | THCA | 갑상선암 |
| LGG | 저등급 신경교종 | THYM | 흉선종 |
| LIHC | 간세포암 | UCEC | 자궁내막암 |

### ⚠️ 라벨 계층 문제 (가장 중요)
- **GBMLGG ⊃ LGG**, **KIPAN ⊃ KIRC**. 같은 환자군이 두 라벨로 나뉘어 있다.
  - IDH1 R132H 변이 환자: GBMLGG 171명, LGG 166명 → 변이 정보만으로는 원리적으로 구분 불가능한 부분이 존재.
  - 베이스라인에서 LGG 78%가 GBMLGG로, KIRC 75%가 KIPAN으로 예측됨.
- 구분 단서 가설: GBMLGG는 GBM(교모세포종)을 포함하므로 **EGFR/PTEN 변이, IDH1 WT 비율**이 높음(GBMLGG EGFR 15% vs LGG 3%). KIPAN은 KIRP/KICH를 포함하므로 **VHL 변이율이 낮음**(KIRC 45% vs KIPAN 30%), MET 변이 4%(KIRP 특징).
- 평가 지표가 Macro F1이면 이 4개 클래스가 점수의 약 15%를 좌우.

## 2. 변이 표기법 (HGVS protein notation)
`[원래 아미노산][위치][바뀐 아미노산]` 형식. 예: `V600E` = 600번째 발린(V)이 글루탐산(E)으로.

| 유형 | 표기 | 의미 | 기능 영향 | train 비율 |
|---|---|---|---|---|
| missense | `R132H` | 아미노산 치환 | 있음 (driver일 수 있음) | 64.6% |
| synonymous | `D623D` | 앞뒤 같은 아미노산 | **거의 없음** (silent) | 26.2% |
| nonsense | `R213*` | 종결코돈 생성 → 단백질 절단 | 큼 (기능 상실) | 5.3% |
| frameshift | `K16fs` | 삽입/결실로 틀 이동 | 큼 (기능 상실) | 3.9% |
| in-frame indel | `91_92NH>KY` | 여러 아미노산 교체 | 있음 | 0.1% |

- 한 유전자에 변이가 여러 개면 공백으로 나열(`Q369* I368N`), 최대 64개.
- **동의 변이는 노이즈일 가능성이 높다.** 단, "변이가 많이 생기는 환자"의 지표로는 여전히 의미 있음 → 기능성 변이 이진화와 전체 변이 카운트를 **둘 다** 만들어 비교.

## 3. Driver vs Passenger, Hotspot
- **Driver**: 암을 실제로 일으키는 변이 (TP53, IDH1, BRAF, APC, VHL, PTEN, PIK3CA…). 암종 특이성이 강함.
- **Passenger**: 우연히 생긴 변이. 긴 유전자(RYR2, SYNE1, PCLO)일수록 우연히 많이 맞음 → 변이율이 높아도 정보량은 낮을 수 있음. 단, 변이 부담이 큰 암종(SKCM, LUSC)의 지표는 됨.
- **Hotspot**: 같은 위치에 반복되는 driver 변이. 암종 특이성이 유전자 단위보다 더 높다.
  - IDH1 R132H → 신경교종(LGG/GBMLGG) 337명, LAML 5명
  - BRAF V600E → THCA 184, SKCM 106, COAD 20
  - PIK3CA E545K/H1047R → BRCA, UCEC, HNSC
  - TP53 hotspot(R175H, R273C/H, R248Q/W) → 여러 암종 공통

### 암종별 대표 driver 변이율 (train, %)
| 클래스 | 표지 유전자 |
|---|---|
| LGG | IDH1 79, TP53 51, ATRX 36 |
| GBMLGG | IDH1 41, TP53 37, ATRX 21, PTEN 16, EGFR 15 |
| THCA | BRAF 57 (그 외 거의 없음, 변이 극소) |
| COAD | APC 72, TP53 52 |
| KIRC | VHL 45 / KIPAN | VHL 30 |
| UCEC | PTEN 62, PIK3CA 49, CTNNB1 31 |
| OV | TP53 75 (다른 변이 거의 없음) |
| LUSC | TP53 71 / HNSC | TP53 66 / PAAD | TP53 66 |
| LAML | NPM1 26, IDH1 11, IDH2 11 |
| LIHC | CTNNB1 24, TP53 26 |
| SKCM | BRAF 50 + 초고변이 |
| TGCT | KIT 15 / PRAD | SPOP 11 |
| BRCA | PIK3CA 32, TP53 28, GATA3 7, CDH1 9 |

## 4. ⚠️ 이 패널에 없는 유명 driver 유전자
컬럼 4,384개는 전체 유전자(~20,000)의 일부다. **KRAS, NRAS, TTN, ARID1A, PBRM1, SMAD4, STK11, KEAP1, FLT3, DNMT3A, TET2, BAP1, SETD2, CIC, FUBP1, TERT, FOXA1, MUC16** 등이 없다.
- 영향: PAAD(췌장암, KRAS 90%↑)·LUAD(KRAS 30%)·LAML(FLT3/DNMT3A)·KIRC(PBRM1/BAP1/SETD2)·LGG(CIC/FUBP1)의 핵심 단서가 빠져 있음 → 이 클래스들이 어려운 이유.
- 외부 유전자 지식을 갖다 쓸 수는 없고, 남은 유전자 조합으로 간접 신호를 찾아야 한다.

## 5. 변이 부담(TMB)과 특수 상황
- **Tumor Mutational Burden**: 샘플당 변이 수. 클래스별 중앙값이 THYM 2 ~ SKCM 85로 극단적으로 다름 → 그 자체가 강력한 피처.
- **초과변이(hypermutated) 샘플 101개**(변이 유전자 >300): SKCM 28, STES 27, UCEC 17, COAD 12. 원인은 MSI(현미부수체 불안정, MLH1/MSH2 관련) 또는 POLE 변이(UCEC/COAD). 이 샘플은 이진 피처 수천 개가 1이 되어 트리 모델을 교란할 수 있음 → log 변환, 비율 피처, 혹은 별도 처리.
- **변이 0개 샘플 94개**: THYM 30, THCA 18, LAML 12. 정보가 없어 카운트 피처와 사전확률로만 분류 가능. 이 샘플들은 "변이가 없다"는 사실 자체가 THYM/THCA 신호.

## 6. 분석 시 주의할 함정
1. **test 데이터 절대 미사용** (프로젝트 규칙). 피처 선택, 타깃 인코딩, 희귀 변이 필터 기준을 정할 때도 train fold 내부에서만.
2. **Fold 내 fit**: 타깃 인코딩·빈도 필터·피처 선택은 반드시 각 fold의 train 부분으로만 fit. OOF 점수가 부풀려지지 않도록.
3. **불균형**: BRCA 786 vs DLBC 38. Stratified K-Fold 필수. Macro F1이면 소수 클래스 recall이 점수를 좌우.
4. **희소성**: 0/1 행렬 6,201×4,230, 밀도 0.8%. 트리 모델은 `feature_fraction`을 낮추고, 선형 모델은 scipy sparse로.
5. **긴 유전자 편향**: 변이율이 높다고 중요한 유전자가 아님. 클래스 간 변이율 **차이**(카이제곱, mutual information)로 봐야 함.
6. **다중 변이 셀**: 문자열 그대로 카테고리로 쓰면 10만 개 이상의 유니크 값 → 반드시 파싱해서 구조화.

## 7. 피처 엔지니어링 아이디어 (우선순위)
| 순위 | 피처 | 근거 |
|---|---|---|
| 1 | 유전자별 **기능성 변이** 이진(동의 제외) | 노이즈 제거 |
| 1 | 샘플 단위 카운트: 총/기능성/동의/nonsense/fs 개수, log, 비율 | TMB가 강력 |
| 2 | **hotspot 변이 one-hot** (train에서 빈도 ≥10인 유전자+변이 조합) | 유전자보다 특이성 높음 |
| 2 | 유전자별 nonsense+frameshift 이진 (종양억제유전자 기능 상실 신호: TP53, APC, PTEN, VHL, RB1) | 생물학적 의미 |
| 3 | 유전자 변이 개수(다중 변이 셀 → 정수) | 긴 유전자/초과변이 구분 |
| 3 | 초과변이 플래그, 변이 0개 플래그 | 특수 그룹 |
| 4 | 클래스 판별력 기준 상위 유전자만 선택 (fold 내 카이제곱) | 차원 축소 |
| 4 | 유전자 세트/경로 단위 집계 (예: PI3K 경로, RTK 경로, DNA 복구) | 외부 지식 필요, 후순위 |
