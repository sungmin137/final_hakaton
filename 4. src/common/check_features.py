"""등록 자가 점검: 피처 종류 k의 cw_ 열 수가 v4와 같은지(골격 누락 방지). run_cv.sh가 호출. 1,500행 사용(300행이면 내부 OOF에서 클래스 누락 오류)."""
import sys, warnings; warnings.filterwarnings("ignore")
from main import load_train, FeatureMaker
k = sys.argv[1]; tr = load_train().iloc[:1500]
a = sum(c.startswith("cw_") for c in FeatureMaker("v4").fit(tr).transform(tr).columns)
b = sum(c.startswith("cw_") for c in FeatureMaker(k).fit(tr).transform(tr).columns)
print("OK" if a == b else f"MISSING cw_ {b}/{a}")
