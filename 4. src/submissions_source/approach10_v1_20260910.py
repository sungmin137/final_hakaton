"""접근10 제출 재현 스크립트.

V4 + train OOF 클래스 보정 + 혼동 암종 쌍 재판기 + 쌍둥이 규칙 변형본을 만든다.
실행 시에만 test.csv를 읽는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import approach10_pair_referee_submit

approach10_pair_referee_submit.main()
