# 접근 26 — raw-data 신규 정보축 최종 탐색 (전부 NO-GO)

## 개요
접근16(v19, 팀 최고 0.4799~0.4803)이 쓰고 있는 정보 외에, **train.csv 원본에서 아직 안 쓴 독립적인 정보축이 남아있는지**를 마지막으로 훑은 실험 묶음임. 접근24(NB 앙상블 보강)와 마찬가지로 독립 파이프라인이 아니라 v19에서 파생한 탐색임. 코드 폴더: `6. experiments/2026-09-16_khs_approach26_rawdata_axis/`. 작업일: 2026-09-16, 다른 세션(스크래치 워크스페이스)에서 진행 후 이 문서로 복구·정리함.

**컴플라이언스**: 전 스크립트 train.csv만 사용, test.csv 미사용 확인함(`multi_token_same_mixed.py`, `pmi_gene_embedding.py` 등 전체 grep 확인, 매칭은 전부 "test.csv 미사용" 명시 docstring뿐). 정직 CV(`StratifiedGroupKFold`, group_twins) 유지함.

## 결과 요약 (6개 축 전부 NO-GO)

| # | 축 | 방법 | 결과 | 판정 |
|---|---|---|---|---|
| 1 | EG/max_tok scalar | 유전자당 변이 증거량(EG)·셀당 최대 토큰 수(max_tok)를 burden으로 잔차화(residual)해서 스칼라 피처로 추가 | `step2_overall.csv`: BASE 0.530556 → EGresid −0.0029 / maxtokresid −0.0101 / 둘 다 −0.0020, 전부 하락 | ❌ NO-GO |
| 2 | within-cell 관계 (gene-level NB) | 한 셀 안에 여러 변이가 동시에 적힌 경우(예: `R273C P289L`)까지 구분하는 Multinomial NB로 유전자 단위 정보를 재구성 | standalone Bernoulli 0.177 / Multinomial 0.270(`genenb_result.json`) — 골격(0.53대) 대비 크게 낮음. 앙상블 결합 시 swap −0.0200, add +0.0025~0.0005, seed2718 재검증 swap −0.0135 (`ensemble_result.json`, `genenb_seed2718_result.json`) — 방향 불안정 | ❌ NO-GO |
| 3 | gene PMI/SVD 임베딩 | 유전자 동시발생(co-occurrence) 행렬을 PMI로 만들고 SVD로 저차원 임베딩(k=12) 후 피처로 추가 | standalone macro F1 0.144(매우 약함), v4s+embed delta −0.0038, 변경 563행 중 hurt 158 > helped 141 (`pmi_embedding_result.json`) | ❌ NO-GO |
| 4 | domain→XGB 직접 병합 | 단백질 도메인(protein domain) 정보를 원본 열로 그대로 XGB에 병합 | v4s 단독 위에서는 +0.0028로 미세 개선(`domxgb_result.json`)되나, **접근16 전체 앙상블에 넣으면 −0.0012로 반전**(`domxgb_ensemble_result.json`) — 부분 개선이 전체에선 상쇄 | ❌ NO-GO |
| 5 | gene×class-boundary 스캔 | GBMLGG/LGG, KIPAN/KIRC 경계에서 기존 문헌 드라이버(189개)·기존 조합 피처에 없는 "새 유전자"가 있는지 5-fold 안정성 기준으로 스캔(`boundary_analysis.py`) | PTEN·LRIG1·APC·MAP3K1 등 후보 발굴 — 이 결과가 #6(gene×multi-event)의 입력이 됨. 스캔 자체는 GO/NO-GO 판정 대상이 아니라 #6의 전 단계 | → #6으로 이어짐 |
| 6 | gene×multi-event 조건부 (PTEN/LRIG1/APC/MAP3K1) | #5에서 나온 4개 유전자의 "한 셀에 변이 2개 이상(multi-event)" 여부를 indicator 피처로 v4s에 추가 | 결합 Δ = **−0.000466**(사실상 0, net rescue −15), 개별 ablation 4개 전부 하락(−0.0020~−0.0029). LRIG1·MAP3K1은 fold별 F1이 소수점 4자리까지 완전 동일 — 표본(28~33건)이 너무 희소해 XGBoost가 단 한 번도 유효 분기를 못 만듦(`packed4_v4s_result.json`) | ❌ NO-GO |

## 오늘 raw-data 축 탐색 최종 지도 (세션에서 정리된 결론 그대로)
```
raw-data 축 탐색
→ EG/max_tok scalar        ❌
→ within-cell 관계         ❌
→ gene PMI/SVD 임베딩      ❌
→ domain→XGB 직접 병합     ❌
→ gene×class-boundary      ❌ (burden 교란)
→ gene×multi-event 조건부  ❌ (Stage0 생존 → 최종 OOF에서 NO-GO)
```
모든 후보가 결국 세 가지 중 하나로 수렴함: **(a) burden(변이 개수) 교란, (b) 기존 gene identity 정보의 재확인(중복), (c) 표본 부족**.

## 참고 — Raw Jaccard KNN (별도 세션, 파일 미확보)
이전에 공유받은 정리표에 있던 "Raw Jaccard KNN"(변이 유전자 집합의 유사도 기반 최근접 분류, 31–100구간 macro F1 0.178 vs 기존 0.436으로 크게 열세) 항목은 **다른 스크래치 세션에서 진행된 것으로 추정되나 원본 스크립트를 이번 정리에서 찾지 못함** — 대화 기록상의 수치만 참고로 남김, 파일 근거 없음.

## 결론
접근24(NB 앙상블 보강)와 이번 접근26(raw-data 신규축)을 합쳐 총 14개 시도가 전부 NO-GO로 닫힘. 두 축의 공통 결론은 같음 — **접근16(v19)이 train.csv에서 뽑아낼 수 있는 판별 정보를 이미 대부분 흡수했고, 남은 후보들은 노이즈·중복·표본 부족 중 하나로 수렴함.** "train 원본에서 접근16이 아직 안 쓴 독립적인 정보축을 오늘은 찾지 못했다"가 정직한 결론임.
