"""3단계: 고른 설정(변수세트·선택방법·k·모델)으로 최종 모델을 만들고 해석 결과를 저장.

  - 전체 데이터로 변수 선택 + 부트스트랩 선택 안정성
  - 반복 CV 성능 (변수 고정 → 낙관적 추정. 정직한 추정은 run_selection2 결과)
  - SHAP (mean|SHAP|, 방향, 의존 데이터), CV permutation 중요도
  - 표준화 로지스틱/OLS 계수 (1 SD 증가당 OR 또는 logit 피해도 변화, 95% CI)

사용법: python run_final.py <엑셀> <cls|reg|sev3> <set> <strategy> <k> <model> <출력폴더>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_selection import score, target  # noqa: E402
from run_selection2 import rank_rfe, rank_rfshap, select_boruta, variants  # noqa: E402


def select(X, y, task, strategy, k, seed=0):
    if strategy == "none":
        return list(X.columns)
    if strategy == "boruta":
        return select_boruta(X, y, task, seed)[0]
    rk = {"meanrank": lambda: C.rank_features(X, y, task, seed).index,
          "rfshap": lambda: rank_rfshap(X, y, task, seed),
          "rfe": lambda: rank_rfe(X, y, task, seed)}[strategy]()
    return list(rk[:k])


def fit_pipeline(X, y, task, strategy, k, model, seed=0):
    cols = select(X, y, task, strategy, k, seed)
    m = variants(task, seed)[model].fit(X[cols], y)
    return cols, m


def predict(m, X, task):
    return m.predict_proba(X)[:, 1] if task == "cls" else m.predict(X)


def tree_shap(m, X, task):
    est, Xi = m[-1], m[:-1].transform(X)
    if hasattr(est, "coef_"):          # 선형모형: 계수 × 표준화값
        return np.asarray(Xi) * np.ravel(est.coef_)
    sv = np.asarray(shap.TreeExplainer(est).shap_values(Xi))
    return sv[..., 1] if sv.ndim == 3 else sv


def boot_freq(X, y, task, strategy, k, n_boot=40):
    def one(b):
        idx = np.random.default_rng(1000 + b).integers(0, len(y), len(y))
        return select(X.iloc[idx].reset_index(drop=True), y[idx], task, strategy, k, b)
    sel = Parallel(n_jobs=4)(delayed(one)(b) for b in range(n_boot))
    return pd.Series([c for s in sel for c in s]).value_counts() / n_boot


def std_glm(X, y, task):
    Z = X.fillna(X.median())
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    Z = Z.loc[:, Z.std() > 0]
    exog = sm.add_constant(Z)
    try:
        if task == "cls":
            r = sm.Logit(y, exog).fit(disp=0, method="bfgs", maxiter=2000)
        else:
            r = sm.OLS(y, exog).fit()
    except Exception as e:  # 완전분리 등
        return {"error": str(e)}
    ci = r.conf_int()
    out = pd.DataFrame({"coef": r.params, "lo": ci[0], "hi": ci[1], "p": r.pvalues}).drop("const")
    if task == "cls":
        out[["OR", "OR_lo", "OR_hi"]] = np.exp(out[["coef", "lo", "hi"]])
    fit = {"pseudoR2": r.prsquared} if task == "cls" else {"R2": r.rsquared, "adjR2": r.rsquared_adj}
    return {"table": out.round(4).reset_index(names="var").to_dict("records"), **fit}


def main(path, task, sname, strategy, k, model, outdir):
    k = int(k)
    d = C.load(path)
    dd, y, mtask = target(d, task)
    X = dd[C.feature_sets(dd, stage2=(task != "cls"))[sname]]
    cols, m = fit_pipeline(X, y, mtask, strategy, k, model)
    print("selected", cols)
    freq = boot_freq(X, y, mtask, strategy, k)

    # 변수 고정 반복 CV + fold 별 permutation 중요도
    cv = (RepeatedStratifiedKFold if mtask == "cls" else RepeatedKFold)(
        n_splits=5, n_repeats=10, random_state=7)
    splits = list(cv.split(X, y if mtask == "cls" else np.zeros(len(y))))
    scoring = "roc_auc" if mtask == "cls" else "r2"

    def fold(i, tr, te):
        mm = variants(mtask, i)[model].fit(X.iloc[tr][cols], y[tr])
        pi = permutation_importance(mm, X.iloc[te][cols], y[te], scoring=scoring,
                                    n_repeats=10, random_state=i)
        return te, predict(mm, X.iloc[te][cols], mtask), pi.importances_mean

    res = Parallel(n_jobs=4)(delayed(fold)(i, tr, te) for i, (tr, te) in enumerate(splits))
    oof, scores = {}, []
    for i, (te, p, _) in enumerate(res):
        oof.setdefault(i // 5, np.full(len(y), np.nan))[te] = p
    for p in oof.values():
        scores.append(score(task, y, p))
    perf = pd.DataFrame(scores).agg(["mean", "std"]).round(4).to_dict()
    perm = pd.DataFrame([r[2] for r in res], columns=cols).agg(["mean", "std"]).T

    sv = tree_shap(m, X[cols], mtask)
    Xf = X[cols].fillna(X[cols].median())
    shap_tab = pd.DataFrame({
        "mean_abs_shap": np.abs(sv).mean(0),
        "direction_rho": [spearmanr(Xf[c], sv[:, j]).correlation if Xf[c].nunique() > 1 else 0
                          for j, c in enumerate(cols)],
        "perm_mean": perm["mean"], "perm_std": perm["std"],
        "boot_freq": freq.reindex(cols).fillna(0).values,
    }, index=cols).sort_values("mean_abs_shap", ascending=False)
    print(shap_tab.round(4).to_string())
    print("perf (fixed features, optimistic):", perf)

    tag = f"{task}_final"
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(sv, columns=cols).to_csv(f"{outdir}/{tag}_shap_values.csv", index=False)
    Xf.to_csv(f"{outdir}/{tag}_X.csv", index=False)
    json.dump({
        "task": task, "set": sname, "strategy": strategy, "k": k, "model": model,
        "n": int(len(y)), "selected": cols, "perf_fixed": perf,
        "importance": shap_tab.round(4).reset_index(names="var").to_dict("records"),
        "boot_freq_all": freq.round(3).to_dict(),
        "glm": std_glm(X[cols], y, mtask),
        "oof_mean": np.nanmean(np.vstack(list(oof.values())), 0).round(4).tolist(),
        "y": np.asarray(y).round(4).tolist(),
    }, open(f"{outdir}/{tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:8])
