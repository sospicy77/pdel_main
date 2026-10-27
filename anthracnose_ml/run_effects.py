"""5단계: 정량적 효과 크기 (해석용 통계모형).

ML 최종모형에서 고른 변수 중 상관이 큰 변수(VIF↑)를 군집별 대표 1개로 줄여
  - 발생: 로지스틱 회귀 → 오즈비(OR). 연속형은 1 SD 증가당, 이진형은 0→1
  - 피해도 3단계: 비례오즈 순서형 로지스틱 → OR (한 단계 더 심해질 오즈)
를 추정하고, 같은 변수로 반복 CV 성능도 함께 기록한다.

사용법: python run_effects.py <엑셀> <출력폴더> [all|noweather]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# 발생: 기상 군집(평균기온·습도·IR) 과 방제 군집(살포일수·7월살포) 에서 대표 1개씩
CLS_CORE = ["7~8월_누적강수량", "노출값", "6월_평균기온", "7~8월_사후지연_최대",
            "7~8월_사전경과_최대", "작년_탄저발생", "제초매트", "log_성묘개수", "보호_카_횟수"]
# 피해도 3단계: 최종 XGB 모형의 선택 변수 그대로 (VIF 낮음)
SEV_CORE = ["초발_DOY", "이환과_즉시제거", "추비_횟수", "작년_탄저발생", "보호_카_횟수",
            "정식_초발_경과일", "7~8월_사전경과_최대"]


def design(X):
    """연속형은 표준화(1 SD), 이진형은 그대로."""
    X = X.fillna(X.median())
    binary = [c for c in X if X[c].nunique() == 2]
    Z = X.copy()
    for c in X:
        if c not in binary:
            Z[c] = (X[c] - X[c].mean()) / X[c].std()
    sd = {c: (None if c in binary else round(float(X[c].std()), 3)) for c in X}
    return Z, sd


def table(params, ci, p, sd, cols):
    out = pd.DataFrame({"OR": np.exp(params[cols]), "lo": np.exp(ci.loc[cols, 0]),
                        "hi": np.exp(ci.loc[cols, 1]), "p": p[cols]})
    out["unit"] = ["0→1" if sd[c] is None else f"+1SD ({sd[c]})" for c in cols]
    return out.round(4).reset_index(names="var")


def vif(Z):
    A = sm.add_constant(Z).values
    return {c: round(variance_inflation_factor(A, i + 1), 2) for i, c in enumerate(Z)}


# 기상 변수 제외 모형: 기상 정보는 노출값만
CLS_CORE_NW = ["노출값", "7~8월_사후지연_최대", "7~8월_사전경과_최대", "작년_탄저발생", "제초매트",
               "log_성묘개수", "보호_카_횟수"]


def main(path, outdir, variant="all"):
    core = CLS_CORE_NW if variant == "noweather" else CLS_CORE
    d = C.load(path)
    res = {}

    Z, sd = design(d[core])
    y = d[C.Y1].values
    r = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                          Z, y, scoring="roc_auc",
                          cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=3))
    res["cls"] = {"table": table(r.params, r.conf_int(), r.pvalues, sd, core).to_dict("records"),
                  "vif": vif(Z), "pseudoR2": round(r.prsquared, 4),
                  "cv_auc": [round(auc.mean(), 4), round(auc.std(), 4)], "n": int(len(y))}

    if variant == "noweather":
        json.dump(res, open(f"{outdir}/effects_noweather.json", "w"), ensure_ascii=False, indent=1)
        print(pd.DataFrame(res["cls"]["table"]).to_string(), res["cls"]["cv_auc"])
        return
    dd = d[d[C.Y1] == 1].reset_index(drop=True)
    Z, sd = design(dd[SEV_CORE])
    ys = pd.Series(pd.Categorical(pd.cut(dd[C.Y2], C.SEV_BINS, labels=False, right=False),
                                  ordered=True))
    r = OrderedModel(ys, Z, distr="logit").fit(method="bfgs", disp=0, maxiter=5000)
    res["sev3"] = {"table": table(r.params, r.conf_int(), r.pvalues, sd, SEV_CORE).to_dict("records"),
                   "vif": vif(Z), "n": int(len(ys)),
                   "class_counts": ys.value_counts().sort_index().tolist()}
    for k, v in res.items():
        print(k, {a: b for a, b in v.items() if a != "table"})
        print(pd.DataFrame(v["table"]).to_string())
    json.dump(res, open(f"{outdir}/effects.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
