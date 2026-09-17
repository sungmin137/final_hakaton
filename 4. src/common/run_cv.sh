#!/bin/bash
# 정직 CV 실행기 (2026-09-15). 목적: 인자 오류·조용한 실패 방지.
#   사용: bash "4. src/common/run_cv.sh" <features> [params=mild_col]
#   - bash로 고정(zsh 변수 분리 차이 회피), 인자 검증, 시작/종료/결과를 6. experiments/cv_runs.log 에 남김, 실패 시 exit≠0
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT" || exit 2
FEAT="${1:?features 인자 필요}"; PARAMS="${2:-mild_col}"; LOG="$ROOT/6. experiments/cv_runs.log"
CHOICES="$(PYTHONPATH="4. src/common" python3 -c "import re,sys;s=open('4. src/common/main.py').read();print(' '.join(re.search(r'\"--features\".*?choices=\[(.*?)\]',s,re.S).group(1).replace('\"','').replace(',',' ').split()))")"
if ! grep -qw -- "$FEAT" <<<"$CHOICES"; then echo "FAIL $(date '+%F %T') features='$FEAT' 는 선택지에 없음: $CHOICES" | tee -a "$LOG"; exit 3; fi
# 골격 누락 자가 점검 (2026-09-17): v4 골격 계열이면 cw_ 열 수가 v4와 같아야 한다 (check_features.py)
case "$FEAT" in official|v1|a4|a7|a8|spec) ;; *)
  CHK="$(PYTHONPATH="4. src/common" python3 "4. src/common/check_features.py" "$FEAT" 2>/dev/null | tail -1)"
  if [ "$CHK" != "OK" ]; then echo "FAIL $(date '+%F %T') features=$FEAT 골격 누락: $CHK (main.py 등록 목록 확인)" | tee -a "$LOG"; exit 5; fi ;;
esac
echo "START $(date '+%F %T') features=$FEAT params=$PARAMS" | tee -a "$LOG"
OUT="$(PYTHONPATH="4. src/common" python3 "4. src/common/main.py" --features "$FEAT" --cv --group-twins --params "$PARAMS" 2>&1 | grep -v Warning)"; RC=${PIPESTATUS[0]}
RES="$(grep 'OOF macroF1' <<<"$OUT" | tail -1)"
if [ "$RC" -ne 0 ] || [ -z "$RES" ]; then echo "FAIL $(date '+%F %T') features=$FEAT rc=$RC :: $(tail -3 <<<"$OUT" | tr '\n' ' ')" | tee -a "$LOG"; exit 4; fi
echo "OK   $(date '+%F %T') features=$FEAT params=$PARAMS :: $RES" | tee -a "$LOG"
