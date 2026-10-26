"""발병농가 67건 피해도 회귀 성능 개선.

stage A (transform): 목표변수 변환 × 모델 × top-k 비교 (변수 선택은 fold 안, 하이퍼파라미터 기본값)
stage B (tune)     : 고른 변환에서 변수 선택 + 하이퍼파라미터 튜닝을 모두 안쪽 CV 로 하는 nested CV
stage C (subset)   : 후보 변수 풀에서 최적 부분집합(PRESS=LOO 오차 최소)을 fold 안에서 찾는 선형/Huber 회귀

사용법: python run_reg_tune.py <엑셀> <A|B|C> <출력폴더> [변환] [변수세트]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

N = 67
# 변환 f 와 역변환 g (원척도 R² 계산용)
TRANSFORMS = {
    "logit": (C.sev_to_logit, C.logit_to_sev),
    "log10": (lambda y: np.log10(y), lambda z: np.clip(10 ** z, 0, 1)),
    "sqrt": (np.sqrt, lambda z: np.clip(z, 0, 1) ** 2),
    "asin": (lambda y: np.arcsin(np.sqrt(y)), lambda z: np.sin(np.clip(z, 0, np.pi / 2)) ** 2),
    "raw": (lambda y: np.asarray(y, float), lambda z: np.clip(z, 0, 1)),
}
KS = [3, 5, 7, 10, 15, 20, 1000]


def base_models(seed=0):
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    return {
        "EN": make_pipeline(imp, sc, ElasticNetCV(l1_ratio=[.2, .5, .9], cv=5, max_iter=20000)),
        "Huber": make_pipeline(imp, sc, HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=2000)),
        "SVR": make_pipeline(imp, sc, SVR(C=1.0, epsilon=0.1, gamma="scale")),
        "RF": make_pipeline(imp, RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5,
                                                       random_state=seed, n_jobs=1)),
        "ET": make_pipeline(imp, ExtraTreesRegressor(400, min_samples_leaf=3, max_features="sqrt",
                                                     random_state=seed, n_jobs=1)),
        # 튜닝 결과 (log10, k=7) 가장 안정적이었던 설정
        "ET_tuned": make_pipeline(imp, ExtraTreesRegressor(500, min_samples_leaf=2, max_features="sqrt",
                                                           random_state=seed, n_jobs=1)),
        "XGB": make_pipeline(imp, XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                               subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
                                               random_state=seed, n_jobs=1, verbosity=0)),
    }


def tuned_models(seed=0):
    """(파이프라인, 탐색공간). 스케일러 이후 마지막 단계 이름으로 파라미터 지정."""
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    return {
        "Huber": (make_pipeline(imp, sc, HuberRegressor(max_iter=3000)),
                  {"huberregressor__epsilon": uniform(1.05, 1.5),
                   "huberregressor__alpha": loguniform(1e-3, 30)}),
        "SVR": (make_pipeline(imp, sc, SVR()),
                {"svr__C": loguniform(0.05, 50), "svr__epsilon": loguniform(0.01, 1),
                 "svr__gamma": loguniform(1e-3, 0.5)}),
        "ET": (make_pipeline(imp, ExtraTreesRegressor(200, random_state=seed, n_jobs=1)),
               {"extratreesregressor__min_samples_leaf": randint(1, 8),
                "extratreesregressor__max_features": uniform(0.2, 0.8),
                "extratreesregressor__max_depth": [None, 3, 5, 8]}),
        "XGB": (make_pipeline(imp, XGBRegressor(random_state=seed, n_jobs=1, verbosity=0)),
                {"xgbregressor__n_estimators": randint(100, 600),
                 "xgbregressor__max_depth": randint(1, 4),
                 "xgbregressor__learning_rate": loguniform(0.01, 0.2),
                 "xgbregressor__subsample": uniform(0.6, 0.4),
                 "xgbregressor__colsample_bytree": uniform(0.4, 0.6),
                 "xgbregressor__min_child_weight": randint(1, 6),
                 "xgbregressor__reg_lambda": loguniform(0.1, 20)}),
    }


def scores(y, z, pred_z, inv):
    return {"R2_t": r2_score(z, pred_z), "R2_raw": r2_score(y, inv(pred_z)),
            "rho": spearmanr(y, pred_z).correlation}


def data(path, sname):
    d = C.load(path)
    dd = d[d[C.Y1] == 1].reset_index(drop=True)
    return dd, dd[C.feature_sets(dd, stage2=True)[sname]], dd[C.Y2].values


# ---------------- stage A ----------------
def fold_A(X, z, tr, te, seed):
    rk = C.rank_features(X.iloc[tr], z[tr], "reg", seed).index
    out = []
    for k in KS:
        cols = list(rk[:k])
        for m, mod in base_models(seed).items():
            if m == "ET_tuned":
                continue
            mod.fit(X.iloc[tr][cols], z[tr])
            out.append((min(k, X.shape[1]), m, te, mod.predict(X.iloc[te][cols])))
    return out


def stage_A(path, outdir, sname="A"):
    dd, X, y = data(path, sname)
    cv = list(RepeatedKFold(n_splits=5, n_repeats=6, random_state=42).split(X))
    rows = []
    for tname, (f, inv) in TRANSFORMS.items():
        z = f(y)
        res = Parallel(n_jobs=4)(delayed(fold_A)(X, z, tr, te, i) for i, (tr, te) in enumerate(cv))
        preds = {}
        for i, out in enumerate(res):
            for k, m, te, p in out:
                preds.setdefault((k, m, i // 5), np.full(len(y), np.nan))[te] = p
        rows += [{"transform": tname, "k": k, "model": m, "rep": r, **scores(y, z, p, inv)}
                 for (k, m, r), p in preds.items()]
        print(tname, "done", flush=True)
    r = pd.DataFrame(rows)
    s = r.groupby(["transform", "model", "k"]).agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    s = s.sort_values("R2_t_mean", ascending=False)
    s.round(4).to_csv(f"{outdir}/regtune_A_{sname}.csv")
    print(s.groupby("transform").head(3).round(3).to_string())


# ---------------- stage B ----------------
def fold_B(X, z, tr, te, seed, ks=(5, 7, 10, 15)):
    rk = C.rank_features(X.iloc[tr], z[tr], "reg", seed).index
    out = []
    inner = KFold(5, shuffle=True, random_state=seed)
    for m, (pipe, space) in tuned_models(seed).items():
        best = None
        for k in ks:   # k 도 안쪽 CV 로 고른다
            cols = list(rk[:k])
            rs = RandomizedSearchCV(pipe, space, n_iter=15, cv=inner, scoring="r2",
                                    random_state=seed, n_jobs=1).fit(X.iloc[tr][cols], z[tr])
            if best is None or rs.best_score_ > best[0]:
                best = (rs.best_score_, k, cols, rs.best_estimator_, rs.best_params_)
        _, k, cols, est, params = best
        out.append((m, te, est.predict(X.iloc[te][cols]), k, cols, params))
    return out


def stage_B(path, outdir, tname, sname="A"):
    dd, X, y = data(path, sname)
    f, inv = TRANSFORMS[tname]
    z = f(y)
    cv = list(RepeatedKFold(n_splits=5, n_repeats=4, random_state=7).split(X))
    res = Parallel(n_jobs=4)(delayed(fold_B)(X, z, tr, te, i) for i, (tr, te) in enumerate(cv))
    preds, chosen = {}, []
    for i, out in enumerate(res):
        for m, te, p, k, cols, params in out:
            preds.setdefault((m, i // 5), np.full(len(y), np.nan))[te] = p
            chosen.append({"model": m, "fold": i, "k": k, "cols": "|".join(cols),
                           **{kk.split("__")[1]: v for kk, v in params.items()}})
    r = pd.DataFrame([{"model": m, "rep": rep, **scores(y, z, p, inv)} for (m, rep), p in preds.items()])
    s = r.groupby("model").agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    tag = f"{tname}_{sname}"
    s.round(4).to_csv(f"{outdir}/regtune_B_{tag}.csv")
    pd.DataFrame(chosen).to_csv(f"{outdir}/regtune_B_{tag}_chosen.csv", index=False)
    # 앙상블(모델 평균)도 함께
    ens = [np.mean([preds[(m, rep)] for m in tuned_models()], 0) for rep in range(4)]
    print(s.round(3).to_string())
    print("ensemble", pd.DataFrame([scores(y, z, p, inv) for p in ens]).mean().round(3).to_dict())


# ---------------- stage C ----------------
def press_r2(Xm, z):
    """OLS 의 LOO 예측오차(PRESS) 기반 R²."""
    A = np.column_stack([np.ones(len(z)), Xm])
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    e = z - H @ z
    loo = e / np.clip(1 - np.diag(H), 1e-6, None)
    return 1 - (loo ** 2).sum() / ((z - z.mean()) ** 2).sum()


def best_subset(X, z, pool, max_size=6):
    Xs = X[pool].fillna(X[pool].median())
    Xs = ((Xs - Xs.mean()) / Xs.std()).values
    best = (-np.inf, None)
    for s in range(1, max_size + 1):
        for comb in itertools.combinations(range(len(pool)), s):
            r = press_r2(Xs[:, comb], z)
            if r > best[0]:
                best = (r, [pool[i] for i in comb])
    return best


def fold_C(X, z, tr, te, seed, pool_size=14):
    pool = list(C.rank_features(X.iloc[tr], z[tr], "reg", seed).index[:pool_size])
    sc, cols = best_subset(X.iloc[tr], z[tr], pool)
    out = []
    for m, mod in {"OLS": LinearRegression(),
                   "Huber": HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=3000)}.items():
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), mod)
        p.fit(X.iloc[tr][cols], z[tr])
        out.append((m, te, p.predict(X.iloc[te][cols]), cols))
    return out


def stage_C(path, outdir, tname, sname="A"):
    dd, X, y = data(path, sname)
    f, inv = TRANSFORMS[tname]
    z = f(y)
    cv = list(RepeatedKFold(n_splits=5, n_repeats=4, random_state=7).split(X))
    res = Parallel(n_jobs=4)(delayed(fold_C)(X, z, tr, te, i) for i, (tr, te) in enumerate(cv))
    preds, chosen = {}, []
    for i, out in enumerate(res):
        for m, te, p, cols in out:
            preds.setdefault((m, i // 5), np.full(len(y), np.nan))[te] = p
            if m == "OLS":
                chosen.append("|".join(sorted(cols)))
    r = pd.DataFrame([{"model": m, "rep": rep, **scores(y, z, p, inv)} for (m, rep), p in preds.items()])
    s = r.groupby("model").agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    tag = f"{tname}_{sname}"
    s.round(4).to_csv(f"{outdir}/regtune_C_{tag}.csv")
    v = pd.Series([c for s_ in chosen for c in s_.split("|")]).value_counts() / len(chosen)
    v.round(3).to_csv(f"{outdir}/regtune_C_{tag}_varfreq.csv")
    print(s.round(3).to_string())
    print(v.head(12).round(2).to_string())


if __name__ == "__main__":
    path, stage, outdir = sys.argv[1:4]
    extra = sys.argv[4:]
    {"A": lambda: stage_A(path, outdir, *extra),
     "B": lambda: stage_B(path, outdir, *extra),
     "C": lambda: stage_C(path, outdir, *extra)}[stage]()
