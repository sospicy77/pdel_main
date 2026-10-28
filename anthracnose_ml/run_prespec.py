"""효과 크기(오즈비)를 사전 지정 모형으로 추정 + 안정성 선택 빈도를 근거 강도로 보고.

같은 자료로 변수를 고른 뒤 그 변수로 오즈비를 내면 오즈비가 부풀고 p 값·CI 가 타당하지 않으므로,
변수는 취약성 틀(노출·민감도·적응력, 05장)로 미리 정한다.
  noweather: 노출값 + 민감도(작년 발생, 연작, 인근 고추밭) + 적응력(강우 전 방제 경과일, 제초매트)
             + 보정(농가 규모)  → 7개
  all      : 위 7개 + 기상 묶음 대표 2개(7~8월 누적강수량, 6월 평균기온)  → 9개
추정: Firth 보정 로지스틱(사건 수 대비 변수가 많을 때 계수 과대추정을 줄임) + 비교용 최대우도 로지스틱.
근거 강도: 해당 최종모형 후보 변수(25개) 전체에 대한 안정성 선택 빈도.
  크기 n/2 부분표본 500회 × L1 규제 경로, 평균 선택 변수 수 q ≤ 5 인 규제 구간에서의 최대 빈도
  (Meinshausen & Bühlmann 2010; 빈도 기준 π = 0.7 이면 잘못 선택되는 변수 기대 수 ≤ q²/((2π−1)p)).

사용법: python run_prespec.py <labeled 엑셀> <변수세트 폴더> <all|noweather>
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_effects import design, table, vif  # noqa: E402

warnings.filterwarnings("ignore")
E = ["노출값"]
S = ["작년_탄저발생", "연작", "인근고추밭_50m"]
# 적응력: 05장 취약성 분석과 같은 2개 (후보 조합 9개 비교로 선정, run_adapt_compare)
A = ["7~8월_사전경과_최대", "제초매트"]
COV = ["log_성묘개수"]
WEATHER = ["7~8월_누적강수량", "6월_평균기온"]
ROLE = {**{v: "노출" for v in E}, **{v: "민감도" for v in S}, **{v: "적응력" for v in A},
        **{v: "보정" for v in COV}, **{v: "기상" for v in WEATHER}}
SPEC = {"noweather": E + S + A + COV, "all": E + WEATHER + S + A + COV}
Q, PI, N_SUB = 5, 0.7, 500


def firth(X, y, iters=100, tol=1e-8):
    """Firth 보정 로지스틱 (Jeffreys 사전분포 벌점). X 에 상수열 포함."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        XtWX = X.T @ (X * W[:, None])
        inv = np.linalg.inv(XtWX)
        h = W * np.einsum("ij,jk,ik->i", X, inv, X)          # 영향력(hat) 대각
        U = X.T @ (y - p + h * (0.5 - p))
        step = inv @ U
        # 한 번에 너무 크게 움직이지 않도록 제한
        mx = np.abs(step).max()
        if mx > 5:
            step *= 5 / mx
        b = b + step
        if np.abs(step).max() < tol:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    inv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    return b, np.sqrt(np.diag(inv))


def stability(Xc, y, seed=0):
    Z = ((Xc - Xc.mean()) / Xc.std()).values
    Cs = np.logspace(-1.3, 0, 14)
    rng = np.random.default_rng(seed)
    H = np.zeros((len(Cs), Z.shape[1]))
    ns = np.zeros(len(Cs))
    for _ in range(N_SUB):
        i = rng.choice(len(y), len(y) // 2, replace=False)
        for ci, c in enumerate(Cs):
            co = LogisticRegression(penalty="l1", solver="liblinear", C=c).fit(Z[i], y[i]).coef_[0]
            H[ci] += co != 0
            ns[ci] += (co != 0).sum()
    H /= N_SUB
    ns /= N_SUB
    keep = ns <= Q
    freq = pd.Series(H[keep].max(0), index=Xc.columns)
    ev = Q * Q / ((2 * PI - 1) * Xc.shape[1])
    return freq.sort_values(ascending=False), float(Cs[keep].max()), ev


def cv_auc(Z, y, n_rep=10):
    aucs = []
    for rep in range(n_rep):
        P = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=rep).split(Z, y):
            b, _ = firth(sm.add_constant(Z.iloc[tr]).values, y[tr])
            P[te] = 1 / (1 + np.exp(-(sm.add_constant(Z.iloc[te], has_constant="add").values @ b)))
        aucs.append(roc_auc_score(y, P))
    return [float(np.mean(aucs)), float(np.std(aucs))]


def main(xlsx, vdir, variant):
    d = C.load(xlsx)
    y = d[C.Y1].values.astype(float)
    cols = SPEC[variant]
    Z, sd = design(d[cols])
    X = sm.add_constant(Z).values
    b, se = firth(X, y)
    z = norm.ppf(0.975)
    names = ["const"] + cols
    params = pd.Series(b, index=names)
    ci = pd.DataFrame({0: b - z * se, 1: b + z * se}, index=names)
    pv = pd.Series(2 * norm.sf(np.abs(b / se)), index=names)
    t_firth = table(params, ci, pv, sd, cols)
    t_firth.insert(1, "role", [ROLE[c] for c in cols])
    mle = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    t_mle = table(mle.params, mle.conf_int(), mle.pvalues, sd, cols)

    cand = json.load(open(f"{vdir}/cls_final.json"))["selected"]
    cand_all = list(dict.fromkeys(cand + cols))               # 사전 지정 변수가 후보에 없으면 추가
    Xc = d[cand_all].fillna(d[cand_all].median())
    freq, cmax, ev = stability(Xc, y.astype(int))
    out = {"variant": variant, "spec": cols, "roles": {c: ROLE[c] for c in cols},
           "events": int(y.sum()), "n": int(len(y)), "epv": round(y.sum() / len(cols), 2),
           "cls": {"table": t_firth.to_dict("records"), "vif": vif(Z), "pseudoR2": round(mle.prsquared, 4),
                   "cv_auc": cv_auc(Z, y), "n": int(len(y))},
           "mle_table": t_mle.to_dict("records"),
           "stability": {"freq": freq.round(3).to_dict(), "q": Q, "pi": PI, "C_max": cmax, "EV_bound": round(ev, 2),
                         "n_candidates": len(cand_all), "n_sub": N_SUB}}
    json.dump(out, open(f"{vdir}/prespec.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(variant, "EPV", out["epv"], "CV AUC", out["cls"]["cv_auc"], "VIF max", max(out["cls"]["vif"].values()))
    print(t_firth.to_string())
    print("MLE OR:", dict(zip(t_mle["var"], t_mle["OR"])))
    print(freq.head(15).round(2).to_string(), "E[V]<=", round(ev, 2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
