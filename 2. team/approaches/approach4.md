# 접근 4 — 문헌 기반 변이 지식 (Literature Driver Map)

## 개요
논문·교과서 수준의 지식으로 **암종별 알려진 driver 변이 항목**(유전자, 역할 oncogene/TSG, 경로, hotspot 위치, 문헌 빈도)을 표로 만들고,
train 데이터와 대조해 일치·불일치를 분석한 뒤, 그 표에서 직접 계산되는 피처를 모델에 넣는다.
환자 단위 외부 데이터는 쓰지 않는다(2. team/rules_compliance.md 해석). 분석 문서: `3. docs/06_literature_driver_analysis.md`.

코드: `4. src/common/approach4_literature/` — `literature_map.py`(지식 표), `literature_analysis.py`(대조 분석), `literature_features.py`(피처 v5)

## 버전
| 버전 | 파일 (src / submissions) | 바뀐 것 | 정직 CV Macro F1 | LB |
|---|---|---|---|---|
| v1 | (실행 중) | v4 + 문헌 피처 105개 (`--features v5`) | (기입 예정) | - |
| v1-drop | (실행 중) | v1 + test 결측 열 25개 제거 (`--drop-genes`) | (기입 예정) | - |

### v1 — 문헌 피처
- 지식 표: 유전자 189개(역할·경로·hotspot), 암종별 driver 목록 26개(문헌 빈도 포함). 패널에 있는 driver 102개 / 없는 driver 77개(KRAS, ARID1A, EP300, GTF2I, FLT3, DNMT3A, KEAP1, FAT1, CIC…).
- 대조 결과: 문헌 빈도와 관측 변이율은 대부분 ±10%p 안에서 일치 → 같은 TCGA 모집단. 큰 손실은 PAAD(KRAS 90%), UCEC(ARID1A 46%), SKCM(NRAS 28%), LUAD(KRAS 33%).
- 피처: 암종별 문헌 점수 4종×26, 경로 카운트 21, 역할 일치/불일치 4, 여백 2.
- 결측: train NaN 0, test NaN 237셀(25열, 모두 비-driver, 변이율 <2.5%). 기본은 WT 대체, 옵션으로 열 제거.

## 다음 후보
- v2: 문헌 빈도 가중 대신 train에서 추정한 특이도 가중(단, fold 내 fit 필요) / 경로 상호배타성 피처 / TP53 동반 유전자 조합 확장
