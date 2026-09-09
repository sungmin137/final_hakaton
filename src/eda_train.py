"""train.csv만 사용하는 EDA 스크립트. test.csv는 절대 읽지 않는다.
실행: python3 src/eda_train.py
"""
import re
import pandas as pd

TRAIN = "data/raw/train.csv"


def mut_type(s: str) -> str:
    if s.endswith("*"):
        return "nonsense"
    if "fs" in s:
        return "frameshift"
    if "del" in s:
        return "deletion"
    if "ins" in s:
        return "insertion"
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", s)
    if m:
        return "synonymous" if m.group(1) == m.group(3) else "missense"
    return "other"


def main() -> None:
    tr = pd.read_csv(TRAIN)
    genes = tr.columns[2:]
    print("shape", tr.shape)
    print("\n[SUBCLASS 분포]\n", tr["SUBCLASS"].value_counts().to_string())

    sub = tr[genes]
    is_mut = sub != "WT"
    print("\nWT 비율", round(1 - is_mut.values.mean(), 4))
    print("전부 WT인 유전자 수", int((~is_mut).all().sum()))
    print("\n[샘플당 변이 유전자 수]\n", is_mut.sum(axis=1).describe().to_string())

    vals = pd.Series(sub.values.ravel())
    mut = vals[vals != "WT"]
    print("\n다중 변이 셀 비율", round(mut.str.contains(" ").mean(), 4))
    tok = mut.str.split(" ").explode()
    print("\n[개별 변이 유형]\n", tok.map(mut_type).value_counts().to_string())

    print("\n[클래스별 변이수 중앙값]\n",
          is_mut.sum(axis=1).groupby(tr["SUBCLASS"]).median().sort_values().to_string())

    print("\n[클래스별 변이 빈도 상위 5 유전자]")
    for c in sorted(tr["SUBCLASS"].unique()):
        top = is_mut[tr.SUBCLASS == c].mean().sort_values(ascending=False).head(5)
        print(c, {g: round(v, 2) for g, v in top.items()})


if __name__ == "__main__":
    main()
