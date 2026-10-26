"""데이터 버전 비교: v1(150건) → 삭제만 반영(142건, 기존 값) → v2(142건, 삭제+수정).

같은 CV 분할 시드와 같은 파이프라인(변수 선택은 fold 안)으로 성능을 재서,
성능 변화가 행 삭제 때문인지 값 수정 때문인지 나눠 본다.

사용법: python run_compare.py <v1 엑셀> <v2 엑셀> <출력폴더>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_final import fit_pipeline, predict  # noqa: E402
from run_reg_tune import base_models  # noqa: E402

# (과제, 이름, 변수세트, 선택방법, k, 모델)
PIPES = [
    ("cls", "v1 최종: RFE·ET k=20", "B", "rfe", 20, "ET_l3"),
    ("cls", "v2 최종: RFE·ET k=25", "B", "rfe", 25, "ET_l3"),
    ("sev3", "v1 최종: 평균순위·XGB k=7", "A", "meanrank", 7, "XGB"),
    ("sev3", "v2 최종: 평균순위·RF k=5", "A", "meanrank", 5, "RF_l1"),
    ("logit", "v1 최종: logit·ET k=10", "A", "meanrank", 10, "ET_l3"),
    ("log10", "v1 개선: log10·ET(튜닝) k=7", "A", "meanrank", 7, "ET_tuned"),
    ("log10", "v2 최종: log10·EN k=3", "A", "meanrank", 3, "EN"),
]
N_REP = 6


def frames(v1, v2):
    a, b = C.load(v1), C.load(v2)
    deleted = sorted(set(a.ID) - set(b.ID))
    return {"v1 (150건)": a, "삭제만 반영 (142건)": a[a.ID.isin(b.ID)].reset_index(drop=True),
            "v2 (142건)": b}, deleted


def fold(task, X, y, tr, te, sname, st, k, m, seed):
    if task in ("cls", "sev3"):
        mt = "cls" if task == "cls" else "reg"
        cols, mo = fit_pipeline(X.iloc[tr], y[tr], mt, st, k, m, seed)
        return te, predict(mo, X.iloc[te][cols], mt)
    cols = list(C.rank_features(X.iloc[tr], y[tr], "reg", seed).index[:k])
    if m in base_models(seed):
        mo = base_models(seed)[m].fit(X.iloc[tr][cols], y[tr])
    else:
        _, mo = fit_pipeline(X.iloc[tr], y[tr], "reg", "none", 0, m, seed)
        mo.fit(X.iloc[tr][cols], y[tr])
    return te, mo.predict(X.iloc[te][cols])


def evaluate(d, task, sname, st, k, m):
    if task == "cls":
        dd, y = d, d[C.Y1].values
    else:
        dd = d[d[C.Y1] == 1].reset_index(drop=True)
        v = dd[C.Y2].values
        y = {"sev3": pd.cut(v, C.SEV_BINS, labels=False, right=False).astype(float),
             "logit": C.sev_to_logit(v), "log10": np.log10(v)}[task]
    X = dd[C.feature_sets(dd, stage2=(task != "cls"))[sname]]
    cv = (RepeatedStratifiedKFold if task == "cls" else RepeatedKFold)(n_splits=5, n_repeats=N_REP, random_state=21)
    splits = list(cv.split(X, y if task == "cls" else np.zeros(len(y))))
    res = Parallel(n_jobs=4)(delayed(fold)(task, X, y, tr, te, sname, st, k, m, i)
                             for i, (tr, te) in enumerate(splits))
    P = np.zeros((N_REP, len(y)))
    for i, (te, p) in enumerate(res):
        P[i // 5, te] = p
    if task == "cls":
        s = [roc_auc_score(y, p) for p in P]
    elif task == "sev3":
        s = [spearmanr(y, p).correlation for p in P]
    else:
        s = [r2_score(y, p) for p in P]
    return float(np.mean(s)), float(np.std(s)), len(y)


def main(v1, v2, out):
    fr, deleted = frames(v1, v2)
    rows = []
    for task, name, sname, st, k, m in PIPES:
        for dname, d in fr.items():
            mu, sd, n = evaluate(d, task, sname, st, k, m)
            rows.append({"과제": task, "파이프라인": name, "데이터": dname, "n": n, "mean": mu, "std": sd})
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(f"{out}/compare_versions.csv", index=False)
    json.dump({"deleted": deleted}, open(f"{out}/compare_meta.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
