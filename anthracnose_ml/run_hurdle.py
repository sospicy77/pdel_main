"""4단계: 150건 전체 피해도(Y2) 예측 - 허들 모형 vs 단일 회귀 비교.

허들:  Ŷ2 = P(발생|x) × E[피해도|발생, x]
  - 1단계: 발생 분류 (초발 관련 3변수 제외)
  - 2단계: 발병농가 피해도 회귀. 미발생 농가에는 초발 관련 3변수가 없으므로
           예측용 허들의 2단계에서도 3변수를 쓰지 않는다.
단일:  150건 전체에서 logit(Y2) 를 바로 회귀.
모든 변수 선택은 학습 fold 안에서 수행.

사용법: python run_hurdle.py <엑셀> <set> <strat1> <k1> <model1> <strat2> <k2> <model2> <출력폴더>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_final import fit_pipeline, predict  # noqa: E402


def fold(X, y1, y2, tr, te, a, seed):
    s1, k1, m1, s2, k2, m2 = a
    c1, mo1 = fit_pipeline(X.iloc[tr], y1[tr], "cls", s1, k1, m1, seed)
    p = predict(mo1, X.iloc[te][c1], "cls")
    pos = tr[y1[tr] == 1]
    c2, mo2 = fit_pipeline(X.iloc[pos], C.sev_to_logit(y2[pos]), "reg", s2, k2, m2, seed)
    sev = C.logit_to_sev(predict(mo2, X.iloc[te][c2], "reg"))
    # 단일 회귀: 0 포함 전체를 logit 으로 (같은 2단계 설정 사용)
    c3, mo3 = fit_pipeline(X.iloc[tr], C.sev_to_logit(y2[tr]), "reg", s2, k2, m2, seed)
    single = C.logit_to_sev(predict(mo3, X.iloc[te][c3], "reg"))
    return te, p, sev, single


def metrics(y2, pred):
    lg = lambda v: np.log10(np.asarray(v) + 0.001)
    return {"rho": spearmanr(y2, pred).correlation, "R2": r2_score(y2, pred),
            "R2_log": r2_score(lg(y2), lg(pred)), "AUC_occ": roc_auc_score(y2 > 0, pred)}


def main(path, sname, s1, k1, m1, s2, k2, m2, outdir):
    d = C.load(path)
    X = d[C.feature_sets(d, stage2=False)[sname]]
    y1, y2 = d[C.Y1].values, d[C.Y2].values
    a = (s1, int(k1), m1, s2, int(k2), m2)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=11)
    splits = list(cv.split(X, y1))
    res = Parallel(n_jobs=4)(delayed(fold)(X, y1, y2, tr, te, a, i)
                             for i, (tr, te) in enumerate(splits))
    rows, hurdle_oof = [], np.zeros(len(y2))
    for rep in range(6):
        P, S, U = (np.full(len(y2), np.nan) for _ in range(3))
        for te, p, sev, single in res[rep * 5:(rep + 1) * 5]:
            P[te], S[te], U[te] = p, sev, single
        hurdle_oof += P * S / 6
        rows.append({"model": "hurdle", **metrics(y2, P * S)})
        rows.append({"model": "single", **metrics(y2, U)})
        pos = y2 > 0
        rows.append({"model": "hurdle_stage2_on_diseased", "rho": spearmanr(y2[pos], S[pos]).correlation,
                     "R2": r2_score(C.sev_to_logit(y2[pos]), C.sev_to_logit(S[pos]))})
    r = pd.DataFrame(rows).groupby("model").agg(["mean", "std"]).round(4)
    print(r.to_string())
    os.makedirs(outdir, exist_ok=True)
    r.to_csv(f"{outdir}/hurdle_summary.csv")
    json.dump({"config": dict(zip(["s1", "k1", "m1", "s2", "k2", "m2"], a)), "set": sname,
               "summary": {f"{m}|{c}": v for (c, s), col in r.items() if s == "mean"
                           for m, v in col.items()},
               "oof": hurdle_oof.round(4).tolist(), "y2": y2.tolist()},
              open(f"{outdir}/hurdle.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:10])
