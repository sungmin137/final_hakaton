# 접근 10 — KIRC/KIPAN, GBMLGG/LGG 계단식 마커 피처

## 개요
KIRC(신장 투명세포암)/KIPAN(신장암 전체), GBMLGG(교모세포종+저등급 신경교종)/LGG(저등급 신경교종)가 헷갈리는 문제를 VHL·MET·IDH1·EGFR 등 이미 있는 유전자 컬럼의 조합 피처로 풀어보려는 시도임. 배경·데이터 근거는 `04_todo_day2.md` STEP 3-2 참고.

코드 폴더: `4. src/common/approach10_cascade/`

## 버전
| 버전 | 파일 (approach10_cascade/) | 정직 CV Macro F1 | 결과 | LB |
|---|---|---|---|---|
| v1 | approach10_v1_kidney_brain_markers.py | approach1 v2+approach2 v1·v2(코드명 `v4`) 0.4683 → VHL·MET·IDH1·EGFR 조합 컬럼 4개 추가(`--features v7`) 0.4662 | 실패(-0.0021) | 미제출 — CV가 이미 기준(0.4683)보다 나빠져서 제출 근거 없음 |
| v2 | approach10_v2_paad_combo.py | v4 0.4683 → TP53+CDKN2A 조합 컬럼 1개 추가(`--features v8`) 0.4683 | 무시됨(변화 0) | 미제출 — CV 변화 자체가 없어서(개선 근거 0) 제출 근거 없음 |

실험 기록: `6. experiments/2026-09-10_v7_xgb_grp/`, `6. experiments/2026-09-10_v8_xgb_grp/`

## 상세

### v1 — 신장/뇌종양 계단식 마커 (`approach10_v1_kidney_brain_markers.py`, `CascadeFeatures`)
approach1 v2+approach2 v1·v2 통합 피처(코드명 `v4`) 위에 VHL·MET·IDH1·EGFR 존재 여부 조합 컬럼 4개(`kidney_kirc_signal` 등)를 추가함. KIRC·KIPAN·GBMLGG 전부 악화됐고 LGG만 소폭 개선됨(자세한 클래스별 수치는 `04_todo_day2.md` STEP 3-2 참고).

### v2 — PAAD 오분류 검증 (`approach10_v2_paad_combo.py`, `PaadComboFeature`)
PAAD(췌장암)가 OV(난소암)·BRCA(유방암)로 오분류되는 문제 검증용. KRAS(PAAD 핵심 driver, 90%↑)가 패널에 없어서 TP53만으로는 OV/BRCA와 구분 안 됨 → TP53+CDKN2A 동반 변이 컬럼 1개 추가(OV·BRCA엔 이 조합이 전혀 없음을 train.csv에서 확인함). 결과: 완전히 무시됨. PAAD confusion matrix가 v4와 소수점까지 동일함.

## 결론 — 채택하지 않음
두 버전 다 VHL·MET·TP53·CDKN2A가 이미 개별 유전자 원-핫 컬럼(`g_VHL` 등)으로 존재해서, "A 있고 B 없으면" 같은 단순 AND 조합은 XGBoost가 이미 두 번의 분기로 스스로 재현할 수 있는 정보임. 새 컬럼을 얹어도 정보가 아니라 노이즈만 추가돼서 실패한 것으로 판단됨 — approach9의 `ovr_<gene>_any`가 죽었던 것과 같은 원인임.

## 공용 코드 처리 (2026-09-10)
실험 당시 `4. src/common/main.py`의 `FeatureMaker`에 임시로 `v7`/`v8` kind 분기를 추가해서 돌렸으나, 두 버전 다 채택하지 않기로 확정돼서 그 분기를 다시 제거함. 로직 자체는 `4. src/common/approach10_cascade/`에 독립 모듈로 그대로 남겨뒀지만, 팀 공용 파일(`4. src/common/main.py`)에는 실패한 실험의 흔적을 남기지 않음.
