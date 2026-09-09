# 파일 생성 규칙

## 1. 접근법 설명 문서 (md)
- 위치: `2. team/approaches/approachN.md` (N = 접근법 번호)
- **새 접근법을 시도할 때마다 새 md 파일**을 만든다. 파일 하나 = 접근법 하나.
- 내용: 접근법 개요(아이디어, 왜 시도하는지) + **버전별 상세 설명**. 버전이 올라갈 때마다 그 버전 섹션을 추가한다.
- 각 버전 섹션에 반드시 적을 것: 무엇을 바꿨는지(피처·모델·후처리·파라미터), 정직 CV 점수, 리더보드 점수(제출했다면), 만든 py·csv 파일명, 결론(채택/폐기).
- 예시 — `approach1.md`
  > 접근 1: 클래스별 변이 개수 가중치. v1: 개수 가중치 합산 단독 분류기(CV 0.2765). v2: v1 피처 + 가중치 점수 140개를 XGB에 투입, 정직 CV 0.4486, LB 0.41 (`approach1_v2_20260909_1442`).

## 2. 파일 이름
```
approachN_vK_YYYYMMDD_HHMM
```
- `approachN` = 접근법 번호, `vK` = 그 접근법의 버전, 뒤는 **제작 일자와 시각**.
- py 파일(`4. src/submissions/`)과 제출 csv(`5. submissions/`)는 **같은 이름**을 쓴다 → 어떤 코드가 어떤 제출을 만들었는지 1:1 대응.
- 후처리 변형본은 접미사로 구분: `_twin_rule` (쌍둥이 규칙 적용본).
- 버전이 올라가면 새 파일을 만들고, 바뀐 내용은 `2. team/approaches/approachN.md`의 해당 버전 섹션에 상세히 적는다. 기존 파일은 지우지 않는다.

## 3. 코드 (py)
### 3-1. 제출 재현 스크립트 — `4. src/submissions/approachN_vK_YYYYMMDD_HHMM.py`
- **제출 csv 하나 = 스크립트 하나.** 같은 이름의 `5. submissions/….csv`를 만드는 것이 유일한 역할.
- 내용은 "설정"만 담는다: 피처 버전, 파라미터 세트, 후처리 옵션, 출력 이름. 로직은 넣지 않고 `4. src/common/make_submission.py`의 `main()`을 호출한다.
- 파일 머리(docstring)에 반드시: 접근법·버전 한 줄 설명, 재현 명령, 사용한 설정, 정직 CV·LB 점수, 설명 문서(`2. team/approaches/approachN.md`) 경로.
- 실행: 루트에서 `python3 "4. src/submissions/<이름>.py"` → `5. submissions/<이름>.csv` 생성 (+ `submission.csv` 갱신, 쌍둥이 규칙본은 `6. experiments/submissions/twin_rule/`로).
- 새 버전을 만들 때: 기존 스크립트를 복사해 이름(버전·시각)을 바꾸고 설정만 수정한다. 기존 스크립트·csv는 지우지 않는다.
- 목록은 `4. src/submissions/README.md`에 한 줄씩 추가한다.

템플릿:
```python
"""접근 2 v2: 지식 피처 + 클래스 배율 후처리 (정직 CV 0.4691+0.012, LB 0.43)

재현: python3 "4. src/submissions/approach2_v2_20260909_1653.py"  →  5. submissions/approach2_v2_20260909_1653.csv
설정: --features v4 --class-scale 6. experiments/2026-09-09_v4_xgb_grp
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission
sys.argv = ["make_submission.py", "--features", "v4", "--class-scale", "6. experiments/2026-09-09_v4_xgb_grp",
            "--tag", "approach2_v2_20260909_1653"]
make_submission.main()
```

### 3-2. 공용 라이브러리 — `4. src/common/`
- 피처 생성, 정직 CV, 후처리, 모델, 분석 코드. 이름 규칙의 예외(접근법 폴더·모듈명 사용).
- `main.py` 파이프라인(FeatureMaker, cross_validate) · `make_submission.py` 제출 생성(test.csv를 읽는 유일한 지점) · `features/` `approach1_count_weight/` `approach2_knowledge/` `approach3_class_feature_compare/` `postprocess/` `models/` `analysis/`
- 새 접근법의 로직은 `4. src/common/approachN_<이름>/`에 넣고, 진입 스크립트는 그 로직을 옵션으로 호출한다.
- 실행 시 `PYTHONPATH="4. src/common"` (진입 스크립트는 스스로 path를 추가한다).

## 4. 제출 파일 (csv)
- `5. submissions/approachN_vK_YYYYMMDD_HHMM.csv` — `4. src/submissions/`의 py와 동일 이름. `5. submissions/submission.csv`는 가장 최근 파일 복사본.
- `5. submissions/README.md`에 제출 순서·설정·정직 CV·리더보드 점수를 기록한다.

## 5. 재현성
- 진입 스크립트는 같은 입력에서 항상 같은 csv를 만들어야 한다. 실수 계산이 들어가는 피처는 반올림해 고정한다(접근 1 점수: float64 + 소수점 4자리).
- 새 피처를 추가하면 두 번 실행해 csv가 같은지 확인한다.

## 6. 검증·기록
- 모든 실험은 `3. docs/experiments_log.md`에 한 줄 추가 (날짜, 실험명, 설정, 정직 CV, LB, 비고).
- 검증은 반드시 정직 CV(`--group-twins`, 쌍둥이 같은 fold). test.csv는 제출 스크립트의 추론 단계에서만 읽는다.
- 분석 문서(EDA, 도메인, 프로필, 중복 발견, test 관찰)는 `3. docs/`에 둔다.
