"""2단계: 선택한 변수세트에서 변수 선택 방법 × 모델(하이퍼파라미터 변형 포함) × k 비교.

선택 방법(모두 학습 fold 안에서만 수행):
  meanrank : RF·XGB SHAP + 정규화 선형계수의 평균 순위
  rfshap   : RF mean|SHAP| 순위
  rfe      : RF 재귀적 제거 (매 단계 SHAP 하위 15% 제거) 의 제거 순서
  boruta   : 그림자(shuffle) 변수보다 중요도가 유의하게 높은 변수 집합

사용법: python run_selection2.py <엑셀경로> <cls|reg|sev3> <변수세트 A|A_bin|B> <출력폴더>
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binom
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_selection import score, target  # noqa: E402

KS = [3, 5, 7, 10, 15, 20, 25, 1000]
N_SPLITS, N_REPEATS = 5, 6


def variants(task, seed):
    imp = SimpleImputer(strategy="median")
    if task == "cls":
        RF, ET = RandomForestClassifier, ExtraTreesClassifier
        lin, xgb = C.make_model("LR", task, seed), C.make_model("XGB", task, seed)
    else:
        RF, ET = RandomForestRegressor, ExtraTreesRegressor
        lin, xgb = C.make_model("EN", task, seed), C.make_model("XGB", task, seed)
    kw = dict(n_estimators=500, random_state=seed, n_jobs=1)
    return {
        "lin": lin,
        "XGB": xgb,
        "RF_l3": make_pipeline(imp, RF(min_samples_leaf=3, max_features="sqrt", **kw)),
        "RF_l1": make_pipeline(imp, RF(min_samples_leaf=1, max_features="sqrt", **kw)),
        "RF_l5_m5": make_pipeline(imp, RF(min_samples_leaf=5, max_features=0.5, **kw)),
        "ET_l3": make_pipeline(imp, ET(min_samples_leaf=3, max_features="sqrt", **kw)),
    }


def rf_model(task, seed, n=300):
    cls = RandomForestClassifier if task == "cls" else RandomForestRegressor
    return make_pipeline(SimpleImputer(strategy="median"),
                         cls(n_estimators=n, min_samples_leaf=3, max_features="sqrt",
                             random_state=seed, n_jobs=1))


def rank_rfshap(X, y, task, seed):
    m = rf_model(task, seed).fit(X, y)
    return pd.Series(C.shap_abs(m, X, task), index=X.columns).sort_values(ascending=False).index


def rank_rfe(X, y, task, seed):
    cols, removed = list(X.columns), []
    while len(cols) > 1:
        m = rf_model(task, seed, 200).fit(X[cols], y)
        imp = pd.Series(C.shap_abs(m, X[cols], task), index=cols).sort_values()
        drop = list(imp.index[:max(1, int(len(cols) * 0.15))])
        removed += drop
        cols = [c for c in cols if c not in drop]
    return pd.Index(cols + removed[::-1])


def select_boruta(X, y, task, seed, n_iter=20):
    rng = np.random.default_rng(seed)
    Xi = X.fillna(X.median())
    hits = pd.Series(0, index=X.columns)
    for it in range(n_iter):
        sh = Xi.apply(lambda c: rng.permutation(c.values)).add_prefix("sh_")
        m = rf_model(task, seed + it, 300)[-1].fit(pd.concat([Xi, sh], axis=1), y)
        imp = pd.Series(m.feature_importances_, index=list(X.columns) + list(sh.columns))
        hits += (imp[X.columns] > imp[sh.columns].max()).astype(int)
    keep = hits[binom.sf(hits - 1, n_iter, 0.5) < 0.05].index
    if len(keep) < 3:
        keep = hits.sort_values(ascending=False).index[:3]
    return list(keep), hits


def one_fold(X, y, tr, te, task, seed):
    Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
    plans = {
        "meanrank": C.rank_features(Xtr, ytr, task, seed).index,
        "rfshap": rank_rfshap(Xtr, ytr, task, seed),
        "rfe": rank_rfe(Xtr, ytr, task, seed),
    }
    subsets = {}
    for s, rk in plans.items():
        for k in KS:
            if k < len(rk) or k == 1000:
                subsets[(s, min(k, len(rk)))] = list(rk[:k])
    bor, _ = select_boruta(Xtr, ytr, task, seed)
    subsets[("boruta", len(bor))] = bor
    out, chosen = [], {"boruta": bor, "rfe10": list(plans["rfe"][:10]),
                       "rfshap10": list(plans["rfshap"][:10])}
    for (s, k), cols in subsets.items():
        for mname, m in variants(task, seed).items():
            m.fit(Xtr[cols], ytr)
            p = m.predict_proba(Xte[cols])[:, 1] if task == "cls" else m.predict(Xte[cols])
            out.append((s, k, mname, te, p))
    return out, chosen


def main(path, task, sname, outdir):
    d = C.load(path)
    dd, y, mtask = target(d, task)
    X = dd[C.feature_sets(dd, stage2=(task != "cls"))[sname]]
    cv = (RepeatedStratifiedKFold if task == "cls" else RepeatedKFold)(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    splits = list(cv.split(X, y if task == "cls" else np.zeros(len(y))))
    res = Parallel(n_jobs=4)(delayed(one_fold)(X, y, tr, te, mtask, i)
                             for i, (tr, te) in enumerate(splits))
    preds, freq = {}, {}
    for i, (out, chosen) in enumerate(res):
        rep = i // N_SPLITS
        for s, k, m, te, p in out:
            # boruta 는 fold 마다 k 가 달라 "boruta" 하나로 묶음
            key = (s, "var" if s == "boruta" else k, m, rep)
            preds.setdefault(key, np.full(len(y), np.nan))[te] = p
        for s, cols in chosen.items():
            for c in cols:
                freq[(s, c)] = freq.get((s, c), 0) + 1 / len(splits)
    rows = [{"strategy": s, "k": k, "model": m, "rep": rep, **score(task, y, p)}
            for (s, k, m, rep), p in preds.items() if not np.isnan(p).any()]
    r = pd.DataFrame(rows)
    metric = {"cls": "AUC", "reg": "R2", "sev3": "rho"}[task]
    s = r.groupby(["strategy", "model", "k"]).agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    s = s.sort_values(f"{metric}_mean", ascending=False)
    tag = f"{task}_{sname}"
    s.round(4).to_csv(f"{outdir}/{tag}_sel2_summary.csv")
    pd.Series(freq).round(3).sort_values(ascending=False).to_csv(f"{outdir}/{tag}_sel2_freq.csv")
    print(s.head(20).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
