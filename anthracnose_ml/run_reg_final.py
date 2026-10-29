"""최종 피해도 회귀 (반응변수: Y2 피해도_전체, 발병농가만, 초발 관련 3변수 포함).

compare : 반응 척도(log10(Y2), Y2 원값) × 변수세트 × 상위 k × 회귀모델 14종을 5-fold × 10회 CV 로 비교.
          변수 선택(평균순위)은 매 fold 학습자료 안에서만. 상위 모델 평균 앙상블도 같은 fold 로 평가.
nested  : compare 상위 후보 중 (모델, k) 선택까지 안쪽 CV 로 하는 nested CV → 선택 편향 없는 성능.
final   : 고른 설정으로 전체 적합, SHAP·CV permutation 중요도, 농가별 out-of-fold 잔차와 제외 민감도.

사용법: python run_reg_final.py <엑셀> compare|nested|final <출력폴더> [final: 척도 세트 k 모델]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

warnings.filterwarnings("ignore")
KS = [3, 5, 7, 10, 15, 1000]
N_REP = 10
FLOOR = 0.003   # 원값 모델 예측을 log10 로 바꿀 때 하한 (관측 최솟값 0.0036 근처)
SCALES = {"log10": (np.log10, lambda z: np.clip(10 ** z, 0, 1)),
          "raw": (lambda y: np.asarray(y, float), lambda z: np.clip(z, 0, 1))}
FAMILY = {"EN": "선형", "Ridge": "선형", "Huber": "선형", "SVR": "커널", "SVR_b": "커널", "KRR": "커널",
          "GPR": "커널", "kNN": "거리", "RF": "트리", "ET": "트리", "ET_b": "트리", "GBR": "부스팅",
          "XGB": "부스팅", "LGBM": "부스팅", "CatBoost": "부스팅"}


def models(seed=0):
    imp = lambda: SimpleImputer(strategy="median")
    sc = StandardScaler
    gp_kernel = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(0.5)
    return {
        "EN": make_pipeline(imp(), sc(), ElasticNetCV(l1_ratio=[.2, .5, .9], cv=5, max_iter=20000)),
        "Ridge": make_pipeline(imp(), sc(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "Huber": make_pipeline(imp(), sc(), HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=3000)),
        "SVR": make_pipeline(imp(), sc(), SVR(C=1.0, epsilon=0.1, gamma="scale")),
        "SVR_b": make_pipeline(imp(), sc(), SVR(C=3.0, epsilon=0.05, gamma=0.05)),
        # KernelRidge 는 절편이 없어 반응변수를 표준화해서 적합
        "KRR": make_pipeline(imp(), sc(), TransformedTargetRegressor(
            KernelRidge(kernel="rbf", alpha=0.5, gamma=0.05), transformer=StandardScaler())),
        "GPR": make_pipeline(imp(), sc(), GaussianProcessRegressor(gp_kernel, normalize_y=True, random_state=seed)),
        "kNN": make_pipeline(imp(), sc(), KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "RF": make_pipeline(imp(), RandomForestRegressor(500, min_samples_leaf=2, max_features=0.5,
                                                         random_state=seed, n_jobs=1)),
        "ET": make_pipeline(imp(), ExtraTreesRegressor(500, min_samples_leaf=2, max_features="sqrt",
                                                       random_state=seed, n_jobs=1)),
        "ET_b": make_pipeline(imp(), ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.7,
                                                         random_state=seed, n_jobs=1)),
        "GBR": make_pipeline(imp(), GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=2,
                                                              learning_rate=0.03, subsample=0.8,
                                                              random_state=seed)),
        "XGB": make_pipeline(imp(), XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                                                 colsample_bytree=0.7, min_child_weight=2, random_state=seed,
                                                 n_jobs=1, verbosity=0)),
        "LGBM": make_pipeline(imp(), LGBMRegressor(n_estimators=300, num_leaves=7, max_depth=3, learning_rate=0.03,
                                                   min_child_samples=5, subsample=0.8, subsample_freq=1,
                                                   colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)),
        "CatBoost": make_pipeline(imp(), CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                                                           l2_leaf_reg=3, random_seed=seed, verbose=0,
                                                           thread_count=1)),
    }


# 설명변수에서 뺄 변수 (환경변수 REG_EXCLUDE, 쉼표 구분). 예: 초발 시기처럼 피해 진행과 같은 성격이라
# 원인 설명에 쓰면 안 되는 변수. 환경변수라 병렬 작업자 프로세스에도 그대로 전달된다.
EXCLUDE = [v for v in os.environ.get("REG_EXCLUDE", "").split(",") if v]


def data(path, sname):
    d = C.load(path)
    dd = d[d[C.Y1] == 1].reset_index(drop=True)
    cols = [c for c in C.feature_sets(dd, stage2=True)[sname] if c not in EXCLUDE]
    return dd, dd[cols], dd[C.Y2].values


def scores(y, scale, pz):
    pr = SCALES[scale][1](pz)
    return {"R2_log": r2_score(np.log10(y), np.log10(np.clip(pr, FLOOR, 1))),
            "R2_raw": r2_score(y, pr), "RMSE_raw": float(np.sqrt(mean_squared_error(y, pr))),
            "rho": spearmanr(y, pr).correlation}


def fit_predict(name, Xtr, ztr, Xte, seed):
    return models(seed)[name].fit(Xtr, ztr).predict(Xte)


# ---------------- compare ----------------
def fold_compare(X, z, tr, te, seed, names):
    rk = C.rank_features(X.iloc[tr], z[tr], "reg", seed).index
    out = []
    for k in KS:
        cols = list(rk[:k])
        for m in names:
            out.append((min(k, X.shape[1]), m, te, fit_predict(m, X.iloc[tr][cols], z[tr], X.iloc[te][cols], seed)))
    return out


def compare(path, outdir):
    rows, oof_store = [], {}
    for sname in ["A", "B"]:
        dd, X, y = data(path, sname)
        cv = list(RepeatedKFold(n_splits=5, n_repeats=N_REP, random_state=42).split(X))
        for scale, (f, _) in SCALES.items():
            z = f(y)
            res = Parallel(n_jobs=4)(delayed(fold_compare)(X, z, tr, te, i, list(models()))
                                     for i, (tr, te) in enumerate(cv))
            preds = {}
            for i, out in enumerate(res):
                for k, m, te, p in out:
                    preds.setdefault((k, m), np.full((N_REP, len(y)), np.nan))[i // 5, te] = p
            for (k, m), P in preds.items():
                oof_store[(scale, sname, k, m)] = P
                for rep in range(N_REP):
                    rows.append({"scale": scale, "set": sname, "k": k, "model": m, "rep": rep,
                                 **scores(y, scale, P[rep])})
            print(scale, sname, "done", flush=True)
    r = pd.DataFrame(rows)
    s = r.groupby(["scale", "set", "model", "k"]).agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    s = s.sort_values("R2_log_mean", ascending=False).reset_index()

    # 앙상블: 척도·세트별로 서로 다른 모델 상위 3/5개의 out-of-fold 예측 평균
    ens_rows = []
    for (scale, sname), g in s.groupby(["scale", "set"]):
        best = g.drop_duplicates("model")
        y = data(path, sname)[2]
        for n in [3, 5]:
            top = best.head(n)
            P = np.mean([oof_store[(scale, sname, r.k, r.model)] for r in top.itertuples()], 0)
            sc = pd.DataFrame([scores(y, scale, P[rep]) for rep in range(N_REP)])
            ens_rows.append({"scale": scale, "set": sname, "model": f"앙상블(상위{n})", "k": -1,
                             "members": " + ".join(f"{r.model}(k={r.k})" for r in top.itertuples()),
                             **{f"{c}_mean": sc[c].mean() for c in sc}, **{f"{c}_std": sc[c].std() for c in sc}})
    s = pd.concat([s, pd.DataFrame(ens_rows)], ignore_index=True).sort_values("R2_log_mean", ascending=False)
    s.round(4).to_csv(f"{outdir}/reg3_compare.csv", index=False)
    print(s.head(25).round(3).to_string())


# ---------------- nested ----------------
def fold_nested(X, z, y, scale, tr, te, cands, seed):
    inner = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed).split(tr))
    rk_inner = [C.rank_features(X.iloc[tr[a]], z[tr[a]], "reg", seed).index for a, _ in inner]
    best = None
    for k, m in cands:
        P = np.full((2, len(tr)), np.nan)
        for j, (a, b) in enumerate(inner):
            cols = list(rk_inner[j][:k])
            P[j // 5, b] = fit_predict(m, X.iloc[tr[a]][cols], z[tr[a]], X.iloc[tr[b]][cols], seed)
        sc = np.mean([scores(y[tr], scale, p)["R2_log"] for p in P])
        if best is None or sc > best[0]:
            best = (sc, k, m)
    _, k, m = best
    cols = list(C.rank_features(X.iloc[tr], z[tr], "reg", seed).index[:k])
    return te, fit_predict(m, X.iloc[tr][cols], z[tr], X.iloc[te][cols], seed), k, m


def nested(path, outdir, n_cand=10):
    s = pd.read_csv(f"{outdir}/reg3_compare.csv")
    s = s[s.k > 0]
    scale, sname = s.iloc[0][["scale", "set"]]
    g = s[(s.scale == scale) & (s.set == sname)].head(n_cand)
    cands = [(int(r.k), r.model) for r in g.itertuples()]
    dd, X, y = data(path, sname)
    z = SCALES[scale][0](y)
    cv = list(RepeatedKFold(n_splits=5, n_repeats=4, random_state=7).split(X))
    res = Parallel(n_jobs=4)(delayed(fold_nested)(X, z, y, scale, tr, te, cands, i) for i, (tr, te) in enumerate(cv))
    P = np.full((4, len(y)), np.nan)
    chosen = []
    for i, (te, p, k, m) in enumerate(res):
        P[i // 5, te] = p
        chosen.append(f"{m}(k={k})")
    sc = pd.DataFrame([scores(y, scale, p) for p in P])
    out = {"scale": scale, "set": sname, "candidates": [f"{m}(k={k})" for k, m in cands],
           "nested": {c: [float(sc[c].mean()), float(sc[c].std())] for c in sc},
           "chosen_freq": pd.Series(chosen).value_counts().to_dict()}
    json.dump(out, open(f"{outdir}/reg3_nested.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ---------------- final ----------------
def final(path, outdir, scale, sname, k, mname, tag="best"):
    k = int(k)
    dd, X, y = data(path, sname)
    f, inv = SCALES[scale]
    z = f(y)
    cols = list(C.rank_features(X, z, "reg", 0).index[:k])
    m = models(0)[mname].fit(X[cols], z)
    Xf = X[cols].fillna(X[cols].median())

    cv = list(RepeatedKFold(n_splits=5, n_repeats=N_REP, random_state=3).split(X))

    def fold(i, tr, te):
        c = list(C.rank_features(X.iloc[tr], z[tr], "reg", i).index[:k])
        mm = models(i)[mname].fit(X.iloc[tr][c], z[tr])
        pi = permutation_importance(mm, X.iloc[te][c], z[te], scoring="r2", n_repeats=10, random_state=i)
        return te, mm.predict(X.iloc[te][c]), dict(zip(c, pi.importances_mean))

    res = Parallel(n_jobs=4)(delayed(fold)(i, tr, te) for i, (tr, te) in enumerate(cv))
    P = np.full((N_REP, len(y)), np.nan)
    perm, sel = {}, {}
    for i, (te, p, pi) in enumerate(res):
        P[i // 5, te] = p
        for c, v in pi.items():
            perm.setdefault(c, []).append(v)
            sel[c] = sel.get(c, 0) + 1 / len(cv)
    sc = pd.DataFrame([scores(y, scale, p) for p in P])

    # 중요도: 트리모형은 SHAP, 그 외는 표준화 계수(선형) 또는 permutation 만
    est = m[-1]
    shap_abs, direction = None, {}
    if hasattr(est, "feature_importances_") or isinstance(est, CatBoostRegressor):
        sv = np.asarray(shap.TreeExplainer(est).shap_values(m[:-1].transform(X[cols])))
        shap_abs = dict(zip(cols, np.abs(sv).mean(0)))
        direction = {c: float(spearmanr(Xf[c], sv[:, j]).correlation) for j, c in enumerate(cols)}
        pd.DataFrame(sv, columns=cols).to_csv(f"{outdir}/reg3_{tag}_shap.csv", index=False)
        Xf.to_csv(f"{outdir}/reg3_{tag}_X.csv", index=False)
    elif hasattr(est, "coef_"):
        shap_abs = dict(zip(cols, np.abs(np.ravel(est.coef_))))
        direction = dict(zip(cols, np.sign(np.ravel(est.coef_)).astype(float)))
    imp = pd.DataFrame({"perm_mean": {c: np.mean(v) for c, v in perm.items()},
                        "perm_std": {c: np.std(v) for c, v in perm.items()}, "fold_sel": sel})
    if shap_abs:
        imp["mean_abs_shap"] = pd.Series(shap_abs)
        imp["direction"] = pd.Series(direction)
    imp = imp.sort_values("perm_mean", ascending=False)

    # 농가별 잔차 (log10 척도)
    pm = np.mean([np.log10(np.clip(inv(p), FLOOR, 1)) for p in P], 0)
    res_ = np.log10(y) - pm
    rows = pd.DataFrame({"ID": dd.ID, "피해도": y, "예측_피해도": 10 ** pm, "잔차": res_,
                         "SSE기여율": res_ ** 2 / (res_ ** 2).sum(),
                         "이환과_즉시제거": dd["이환과_즉시제거"], "초발_DOY": dd["초발_DOY"],
                         "작년_탄저발생": dd["작년_탄저발생"], "보호_카_횟수": dd["보호_카_횟수"]}
                        ).sort_values("SSE기여율", ascending=False)
    rows.round(4).to_csv(f"{outdir}/reg3_{tag}_rows.csv", index=False)
    sens = []
    for n in [0, 2, 4, 6]:
        keep = ~dd.ID.isin(rows.ID[:n]).values
        Xk, zk, yk = X[keep].reset_index(drop=True), z[keep], y[keep]
        cvk = list(RepeatedKFold(n_splits=5, n_repeats=6, random_state=5).split(Xk))

        def fk(i, tr, te):
            c = list(C.rank_features(Xk.iloc[tr], zk[tr], "reg", i).index[:k])
            return te, fit_predict(mname, Xk.iloc[tr][c], zk[tr], Xk.iloc[te][c], i)

        rr = Parallel(n_jobs=4)(delayed(fk)(i, tr, te) for i, (tr, te) in enumerate(cvk))
        Pk = np.full((6, len(yk)), np.nan)
        for i, (te, p) in enumerate(rr):
            Pk[i // 5, te] = p
        sens.append({"n": n, "ids": list(rows.ID[:n]),
                     "R2_log": float(np.mean([scores(yk, scale, p)["R2_log"] for p in Pk]))})
        print(sens[-1], flush=True)

    out = {"scale": scale, "set": sname, "k": k, "model": mname, "selected": cols,
           "cv": {c: [float(sc[c].mean()), float(sc[c].std())] for c in sc},
           "importance": imp.round(4).reset_index(names="var").to_dict("records"),
           "sensitivity": sens, "oof_log10": pm.round(4).tolist(), "y": y.tolist(), "ids": dd.ID.tolist()}
    if hasattr(est, "coef_"):
        out["coef_std"] = dict(zip(cols, np.round(np.ravel(est.coef_), 4).tolist()))
    json.dump(out, open(f"{outdir}/reg3_{tag}.json", "w"), ensure_ascii=False, indent=1)
    print(imp.round(3).to_string())
    print("cv", out["cv"])
    print(rows.head(12).round(3).to_string())


if __name__ == "__main__":
    path, stage, outdir = sys.argv[1:4]
    {"compare": lambda: compare(path, outdir), "nested": lambda: nested(path, outdir),
     "final": lambda: final(path, outdir, *sys.argv[4:])}[stage]()
