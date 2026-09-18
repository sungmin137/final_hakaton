# 접근 24 — NB 앙상블 보강 축: pairwise referee·burden 가중·position/domain 정보 추가 (전부 NO-GO)

## 개요
독립된 새 파이프라인이 아니라, **접근16(스펙트럼 프로필 NB 파트너, 팀 최고 0.4799)의 확률 출력을 그대로 두고 그 위에 8가지 보강 아이디어를 각각 얹어본 진단성 실험 묶음**임. 성민님의 접근14 v19/v20이 리더보드를 계속 갱신하는 동안, 거기서 파생해서 "더 얹을 수 있는 정보가 남아있는가"를 탐색함. 코드 폴더: `6. experiments/2026-09-15_khs_approach24_nb_referee_routing/`. 작업일: 2026-09-15~16, 다른 세션(스크래치 워크스페이스)에서 진행 후 이 문서로 복구·정리함.

**컴플라이언스**: 전 스크립트 train.csv만 사용, test.csv 미사용 확인함(`nb_burden_deep_dive.py` 등 전체 grep 확인). 정직 CV(`StratifiedGroupKFold`, group_twins) 유지함.

## 결과 요약 (8종 전부 NO-GO)

| # | 시도 | 방법 | 결과 | 판정 |
|---|---|---|---|---|
| 1 | Position × Spectrum | mutation position 정보를 spectrum 피처와 결합 | exact variant의 82%가 singleton(표본 1개)이라 위치 단위 신호가 근본적으로 희소함, position feature 전반 개선 미미 | ❌ NO-GO |
| 2 | NB burden-dependent weighting | mutation burden 구간별로 NB 앙상블 가중치를 다르게 적용 | 후보 5개 전부 flat 가중치(.15) 대비 −0.005~−0.013, 7구간(7-bin) 세분화도 실패함 | ❌ 폐기 |
| 3 | GBMLGG↔LGG pair referee | NB가 이 두 암종 사이에서만 판정을 보정하도록 라우팅 | macro F1 전 구간 악화됨 — GBMLGG는 개선되나 LGG가 크게 손상돼 순손해(`pair_referee_analysis.py`) | ❌ 폐기 |
| 4 | KIPAN↔KIRC pair referee | 동일한 방식의 pairwise referee | ΔF1 −0.00005 ~ −0.0058 (`pair_referee_analysis.py`) | ❌ 폐기 |
| 5 | Asymmetric confidence routing | BASE가 A, NB가 B로 답할 때 특정 방향에서만 confidence 조건부로 결과를 뒤집음 | 방향 제한 로직 자체는 정상 작동하나, 모든 threshold에서 macro F1 악화됨(`asymmetric_routing.py`) | ❌ 폐기 |
| 6 | 397+ mutation deep dive | 초과변이(397개 이상) 구간에서 NB의 높은 확신(margin)이 별도 구조 정보를 주는지 확인 | `deep_dive_out.log` [B-4]: 397+ 구간에서 정답(margin 233.3)과 오답(margin 102.7)의 margin 차이가 통계적으로 유의하지 않음(MWU p=0.19), 오답의 59.1%가 정답 median 이상 margin을 가짐 — "확신 높음=정답"이 성립 안 함. [B-7] 701+ 세부구간은 오히려 정확도 급락(0.538) | ❌ 폐기 |
| 7 | Position-bucket NB | 유전자 상대 위치를 25단위로 나눠 구간 지표(`cwp_bucket_hi_<CLASS>`)를 NB 입력에 추가 | standalone macro F1 0.180(골격 단독 0.53대 대비 크게 낮음), 결합 시 SWAP −0.0089 / ADD −0.0004 (round2/`posnb.py`+`ensemble_a.py`, `ensemble_a_result.json`) | ❌ 폐기 |
| 8 | Protein domain-hit NB | 단백질 도메인(protein domain) 내부 변이 개수를 NB 입력에 추가 | standalone macro F1 0.218, seed42 ADD +0.0022 → seed2718 ADD −0.0005로 재현 안 됨(round2/`domnb.py`+`domnb_seed2718.py`), 기존 hotspot/gene identity 정보와 대부분 중복됨 | ❌ 폐기 |

## 상세 근거 (파일로 직접 확인된 것)

### #7·#8 — position-bucket / domain-hit NB (round2/)
- BASE(접근16 골격) macro F1 = 0.530556 (재구성값, `ensemble_a_result.json`/`ensemble_b_result.json`의 `base_f1`과 일치함)
- posnb(#7) standalone: 0.1799 (`posnb_result.json`) — 골격 단독 대비 압도적으로 낮아, 추가해도 SWAP(대체) −0.0089, ADD(합류) −0.0004
- domnb(#8) standalone: seed42 0.2176 / seed2718 0.2148 (`domnb_result.json`, `domnb_seed2718_result.json`) — seed42에서는 ADD +0.0022로 미세 개선처럼 보이나 seed2718에서 −0.0005로 반전됨, **재현성 없음**으로 폐기

### #6 — 397+ deep dive (`nb_burden_deep_dive.py` → `deep_dive_out.log`)
- burden 구간별 NB 정확도·margin: 1-30(acc 0.247) → 397+(acc 0.639)로 margin은 커지지만, **margin 크기와 정답 여부의 상관이 397+ 구간에서 무너짐**(MWU p=0.19, 유의하지 않음)
- 701+ 세부구간에서 정확도가 오히려 0.538로 급락함 — "변이가 아주 많으면 무조건 신뢰도 높다"는 가정이 깨짐. NB의 높은 확신을 별도 rescue 신호로 쓸 근거 없음

### #3·#4 — pairwise referee (`pair_referee_analysis.py`)
- GBMLGG/LGG, KIPAN/KIRC 모두 이미 twin_rule로 처리되는 쌍둥이 문제와 겹치는 클래스 쌍임. referee가 한쪽을 개선하면 다른 쪽이 그만큼 또는 그 이상 손상되는 제로섬 구조 확인됨 — 접근16 골격이 이미 이 경계에서 최적점 근처에 있다는 방증

### #5 — asymmetric routing (`asymmetric_routing.py`)
- "BASE와 NB 답이 다를 때, 특정 방향(예: BASE→STES를 NB→다른 클래스로)만 confidence 임계값 넘으면 채택" 방식. 방향 제한 로직은 의도대로 작동했으나 모든 threshold 값에서 순손해

## 결론 — 이 축을 여기서 닫는 이유
8개 시도가 공통적으로 수렴한 지점은 세 가지임.
1. **골격(접근16)이 이미 쓸 수 있는 정보를 대부분 흡수**함 — position/domain처럼 원본 열을 그대로 추가하는 방식은 `07_top_submissions_analysis.md`에서 이미 확인된 "원본 열은 트리가 안 쓴다"는 패턴과 같은 결론으로 재확인됨
2. **표본이 얇을수록(position bucket, 397+ 세부구간) 신호가 노이즈에 묻힘**
3. **seed 하나에서 보이는 개선은 재현성 검증(seed2718)에서 절반 가까이 반전됨** — 성급하게 채택했으면 손해였을 항목이 최소 1개(domnb) 있었음

접근24 축은 여기서 종료함. 남은 탐색은 접근26(raw-data 신규 정보축)으로 이어감.
