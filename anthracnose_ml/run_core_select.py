"""효과 크기 해석용 핵심 변수를 재현 가능한 규칙으로 선택하고 로지스틱 오즈비를 추정.

후보: 해당 분류 최종모형(run_final)이 고른 변수 (cls_final.json 의 selected).
규칙
  1. 안정성 선택: 크기 n/2 부분표본 500회, L1 로지스틱(규제 강도 C 는 전체 자료 5-fold CV 로 한 번 결정).
     선택 빈도 ≥ 0.6 인 변수 유지.
  2. 공선성: |Spearman ρ| ≥ 0.7 인 쌍은 빈도가 높은 쪽만 유지 → 이후 VIF > 5 이면 빈도가 가장 낮은 변수부터 제거.
  3. 사건 수 상한: 사건 수 / 5 (EPV ≥ 5) 를 넘으면 빈도 순으로 자름.
결과는 run_effects 와 같은 형식(cls 항목)으로 core_select.json 에 저장.

사용법: python run_core_select.py <labeled 엑셀> <변수세트 폴더(cls_final.json 있는 곳)>
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_effects import design, table, vif  # noqa: E402

warnings.filterwarnings("ignore")
N_SUB, FREQ_MIN, RHO_MAX, VIF_MAX, EPV = 500, 0.6, 0.7, 5.0, 5


def stability(X, y, seed=0):
    Z = ((X - X.mean()) / X.std()).values
    cv = LogisticRegressionCV(Cs=np.logspace(-2, 1, 20), penalty="l1", solver="liblinear", cv=5,
                              scoring="neg_log_loss", random_state=seed).fit(Z, y)
    c = float(cv.C_[0])
    rng = np.random.default_rng(seed)
    hits = np.zeros(X.shape[1])
    signs = np.zeros(X.shape[1])
    for _ in range(N_SUB):
        i = rng.choice(len(y), len(y) // 2, replace=False)
        if y[i].min() == y[i].max():
            continue
        co = LogisticRegression(penalty="l1", solver="liblinear", C=c).fit(Z[i], y[i]).coef_[0]
        hits += co != 0
        signs += np.sign(co)
    freq = pd.Series(hits / N_SUB, index=X.columns)
    sign = pd.Series(signs / np.maximum(hits, 1), index=X.columns)
    return freq, sign, c


def prune(X, freq, cap):
    steps = []
    keep = [v for v in freq.sort_values(ascending=False).index if freq[v] >= FREQ_MIN]
    if len(keep) < 3:                                   # 너무 적으면 빈도 상위 3개
        keep = list(freq.sort_values(ascending=False).index[:3])
        steps.append("빈도 ≥ 0.6 변수가 3개 미만이라 빈도 상위 3개 사용")
    rho = X.corr("spearman").abs()
    for a in list(keep):
        for b in list(keep):
            if a != b and a in keep and b in keep and rho.loc[a, b] >= RHO_MAX:
                drop = a if freq[a] < freq[b] else b
                keep.remove(drop)
                steps.append(f"{drop} 제외: {a if drop == b else b} 와 |ρ| = {rho.loc[a, b]:.2f}")
    while len(keep) > 1:
        Z = sm.add_constant((X[keep] - X[keep].mean()) / X[keep].std())
        v = {c: vif(Z.drop(columns="const"))[c] for c in keep}
        if max(v.values()) <= VIF_MAX:
            break
        drop = min((c for c in keep if v[c] > VIF_MAX), key=lambda c: freq[c])
        keep.remove(drop)
        steps.append(f"{drop} 제외: VIF {v[drop]:.1f}")
    if len(keep) > cap:
        for c in keep[cap:]:
            steps.append(f"{c} 제외: 사건 수 상한 {cap}개 초과 (빈도 {freq[c]:.2f})")
        keep = keep[:cap]
    return keep, steps


def main(xlsx, vdir):
    d = C.load(xlsx)
    y = d[C.Y1].values
    cand = json.load(open(f"{vdir}/cls_final.json"))["selected"]
    X = d[cand].fillna(d[cand].median())
    freq, sign, c = stability(X, y)
    cap = int(y.sum() // EPV)
    core, steps = prune(X, freq, cap)

    Z, sd = design(d[core])
    r = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Z, y,
                          scoring="roc_auc", cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=3))
    out = {"candidates": cand, "C": c, "n_sub": N_SUB, "freq_min": FREQ_MIN, "rho_max": RHO_MAX,
           "vif_max": VIF_MAX, "cap": cap, "events": int(y.sum()),
           "freq": freq.round(3).sort_values(ascending=False).to_dict(), "sign": sign.round(2).to_dict(),
           "steps": steps, "core": core,
           "cls": {"table": table(r.params, r.conf_int(), r.pvalues, sd, core).to_dict("records"),
                   "vif": vif(Z), "pseudoR2": round(r.prsquared, 4),
                   "cv_auc": [round(auc.mean(), 4), round(auc.std(), 4)], "n": int(len(y))}}
    json.dump(out, open(f"{vdir}/core_select.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("C", round(c, 3), "cap", cap)
    print(freq.sort_values(ascending=False).round(2).head(15).to_string())
    print("steps", steps)
    print("core", core)
    print(pd.DataFrame(out["cls"]["table"]).to_string(), out["cls"]["cv_auc"], out["cls"]["pseudoR2"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
