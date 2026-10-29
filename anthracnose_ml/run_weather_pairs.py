"""노출값에 기상 변수 2개를 더하면 발생 모델(피해도 > 3%) 성능이 오르는가.

후보: 기상 8개 + IR면적 2개 중 2개 조합 45개.
1. 선별: 각 조합을 두 모형에 더해 같은 fold(반복 층화 5-fold × 10)로 CV AUC 비교
   a) 04장 기상 제외 최종 RF(RF_l1, 선택 변수 25개 고정) + 조합
   b) 04장 사전 지정 로지스틱(Firth, 7개) + 조합 — 우도비 검정·AIC 도 함께
2. nested: 바깥 CV 학습 fold 안에서 안쪽 CV 로 최고 조합을 고르고 바깥 fold 로 평가
   (45개 중 최고를 고른 낙관을 뺀 개선량)
결과: <출력폴더>/weather_pairs.json, weather_pairs_screen.csv

사용법: python run_weather_pairs.py <labeled 엑셀> <기상 제외 결과 폴더> <출력폴더>
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_effects import design  # noqa: E402
from run_prespec import SPEC, cv_auc, firth  # noqa: E402

warnings.filterwarnings("ignore")
N_REP = 10


def rf(seed, n=500):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=n, min_samples_leaf=1, max_features="sqrt",
                                                random_state=seed, n_jobs=1))


def rf_fold(X, y, tr, te, seed, n=500):
    m = rf(seed, n).fit(X.iloc[tr], y[tr])
    return te, m.predict_proba(X.iloc[te])[:, 1]


def rf_cv(X, y, n_rep=N_REP):
    """반복 CV AUC (반복마다 out-of-fold 예측을 모아 AUC)."""
    aucs = []
    for rep in range(n_rep):
        P = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            P[te] = rf(rep).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, P))
    return float(np.mean(aucs)), float(np.std(aucs))


def logit_stats(d, y, cols, base_ll=None):
    Z, _ = design(d[cols])
    m = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    out = {"AIC": float(m.aic), "ll": float(m.llf), "cv_auc": cv_auc(Z, y)[0]}
    b, se = firth(sm.add_constant(Z).values, y)
    out["firth_OR"] = {c: float(np.exp(b[i + 1])) for i, c in enumerate(cols)}
    out["firth_p"] = {c: float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(b[i + 1] / se[i + 1]))))
                      for i, c in enumerate(cols)}
    if base_ll is not None:
        out["LR_p"] = float(chi2.sf(2 * (m.llf - base_ll), 2))
    return out


def one_pair(d, y, base, pair):
    X = d[base + list(pair)]
    return rf_cv(X, y)


def nested(d, y, base, pairs, n_outer_rep=4):
    """바깥 5-fold × n_outer_rep. 안쪽 5-fold × 2 로 최고 조합 선택(RF 300그루)."""
    rows = []
    for rep in range(n_outer_rep):
        Pb, Pn = np.zeros(len(y)), np.zeros(len(y))
        chosen = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(d, y):
            dt, yt = d.iloc[tr].reset_index(drop=True), y[tr]

            def inner(pair):
                X = dt[base + list(pair)]
                a = []
                for r in range(2):
                    P = np.zeros(len(yt))
                    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=200 + r).split(X, yt):
                        P[ite] = rf(r, 300).fit(X.iloc[itr], yt[itr]).predict_proba(X.iloc[ite])[:, 1]
                    a.append(roc_auc_score(yt, P))
                return np.mean(a)
            sc = Parallel(n_jobs=4)(delayed(inner)(p) for p in pairs)
            best = pairs[int(np.argmax(sc))]
            chosen.append(" + ".join(best))
            Pn[te] = rf(rep).fit(d.iloc[tr][base + list(best)], y[tr]).predict_proba(d.iloc[te][base + list(best)])[:, 1]
            Pb[te] = rf(rep).fit(d.iloc[tr][base], y[tr]).predict_proba(d.iloc[te][base])[:, 1]
        rows.append({"base": roc_auc_score(y, Pb), "nested": roc_auc_score(y, Pn), "chosen": chosen})
        print("nested rep", rep, round(rows[-1]["base"], 3), round(rows[-1]["nested"], 3), chosen, flush=True)
    return rows


def main(xlsx, nwdir, out):
    os.makedirs(out, exist_ok=True)
    d = C.load(xlsx)
    y = d[C.Y1].values.astype(int)
    base = json.load(open(f"{nwdir}/cls_final.json"))["selected"]
    pairs = list(itertools.combinations(C.WEATHER, 2))
    res = {"base_vars": base, "weather": C.WEATHER}

    # 1a) RF
    jobs = [("base", ())] + [(" + ".join(p), p) for p in pairs] + [("기상 10개 전체", tuple(C.WEATHER))]
    rf_res = Parallel(n_jobs=4)(delayed(one_pair)(d, y, base, p) for _, p in jobs)
    # 1b) 사전 지정 로지스틱
    pre = SPEC["noweather"]
    lb = logit_stats(d, y.astype(float), pre)
    res["prespec_base"] = lb
    rows = []
    for (name, p), (m, s) in zip(jobs, rf_res):
        row = {"pair": name, "RF_AUC": m, "RF_AUC_sd": s}
        if 0 < len(p) <= 2:
            ls = logit_stats(d, y.astype(float), pre + list(p), base_ll=lb["ll"])
            row.update(logit_AUC=ls["cv_auc"], AIC=ls["AIC"], LR_p=ls["LR_p"],
                       **{f"OR_{i + 1}": ls["firth_OR"][v] for i, v in enumerate(p)},
                       **{f"p_{i + 1}": ls["firth_p"][v] for i, v in enumerate(p)},
                       OR_노출값=ls["firth_OR"]["노출값"], p_노출값=ls["firth_p"]["노출값"])
        elif name == "base":
            row.update(logit_AUC=lb["cv_auc"], AIC=lb["AIC"], OR_노출값=lb["firth_OR"]["노출값"],
                       p_노출값=lb["firth_p"]["노출값"])
        rows.append(row)
        print(name, round(m, 3), row.get("logit_AUC"), row.get("AIC"), flush=True)
    sc = pd.DataFrame(rows)
    sc.to_csv(f"{out}/weather_pairs_screen.csv", index=False)
    res["screen"] = sc.to_dict("records")

    # 2) nested
    res["nested"] = nested(d, y, base, pairs)
    json.dump(res, open(f"{out}/weather_pairs.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("PAIRSDONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
