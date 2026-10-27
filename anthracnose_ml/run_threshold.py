"""피해도 임계값으로 다시 정한 발생 라벨(Y2 > t)에 대한 분류모형 탐색.

screen : t ∈ {0, 0.005, ..., 0.03} 마다 변수세트(A, B) × 선택방법(평균순위, RFE) × k × 모델을
         반복 층화 5-fold 로 비교 (변수 선택은 fold 안). AUC 외에 PR-AUC, 균형정확도, Brier 도 기록.
label  : 고른 임계값으로 Y1 열을 바꾼 엑셀을 만든다 (이후 run_selection2 / run_final / run_effects 에 그대로 사용).

사용법: python run_threshold.py <엑셀> screen <출력폴더>
        python run_threshold.py <엑셀> label <임계값> <출력 엑셀>
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_selection2 import rank_rfe, variants  # noqa: E402

THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
KS = [5, 10, 15, 20, 25, 1000]
MODELS = ["lin", "RF_l1", "ET_l3", "XGB"]
N_SPLITS, N_REP = 5, 5


def relabel(d, t):
    d = d.copy()
    d[C.Y1] = (d[C.Y2] > t).astype(int)
    return d


def fold(X, y, tr, te, seed):
    plans = {"meanrank": C.rank_features(X.iloc[tr], y[tr], "cls", seed).index,
             "rfe": rank_rfe(X.iloc[tr], y[tr], "cls", seed)}
    out = []
    for st, rk in plans.items():
        for k in KS:
            cols = list(rk[:k])
            for m in MODELS:
                mo = variants("cls", seed)[m].fit(X.iloc[tr][cols], y[tr])
                out.append((st, min(k, X.shape[1]), m, te, mo.predict_proba(X.iloc[te][cols])[:, 1]))
    return out


def screen(path, outdir):
    d0 = C.load(path)
    rows = []
    for t in THRESHOLDS:
        d = relabel(d0, t)
        y = d[C.Y1].values
        for sname in ["A", "B"]:
            part = f"{outdir}/thr_part_{t}_{sname}.csv"
            if os.path.exists(part):   # 중단 후 재실행 시 끝난 조합은 건너뜀
                rows += pd.read_csv(part).to_dict("records")
                continue
            X = d[C.feature_sets(d, stage2=False)[sname]]
            cv = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REP, random_state=42).split(X, y))
            res = Parallel(n_jobs=4)(delayed(fold)(X, y, tr, te, i) for i, (tr, te) in enumerate(cv))
            preds = {}
            for i, out in enumerate(res):
                for st, k, m, te, p in out:
                    preds.setdefault((st, k, m), np.full((N_REP, len(y)), np.nan))[i // N_SPLITS, te] = p
            new = []
            for (st, k, m), P in preds.items():
                for rep, p in enumerate(P):
                    new.append({"threshold": t, "n_pos": int(y.sum()), "prev": y.mean(), "set": sname,
                                 "strategy": st, "k": k, "model": m, "rep": rep,
                                 "AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
                                 "BalAcc": balanced_accuracy_score(y, p >= y.mean()),
                                 "Brier": brier_score_loss(y, p)})
            pd.DataFrame(new).to_csv(part, index=False)
            rows += new
            print(t, sname, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(f"{outdir}/thr_screen_raw.csv", index=False)
    s = r.groupby(["threshold", "n_pos", "prev", "set", "strategy", "model", "k"]).agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    s = s.reset_index().sort_values("AUC_mean", ascending=False)
    s.round(4).to_csv(f"{outdir}/thr_screen_summary.csv", index=False)
    best = s.groupby("threshold").head(1).sort_values("threshold")
    print(best.round(3).to_string())


def label(path, t, out):
    t = float(t)
    x = pd.ExcelFile(path)
    sheets = {s: pd.read_excel(x, s) for s in x.sheet_names}
    d = sheets["모델링_데이터셋"]
    d[C.Y1] = (d[C.Y2] > t).astype(int)
    with pd.ExcelWriter(out) as w:
        for s, df in sheets.items():
            df.to_excel(w, sheet_name=s, index=False)
    print("wrote", out, "positives", int(d[C.Y1].sum()))


if __name__ == "__main__":
    path, stage = sys.argv[1:3]
    {"screen": lambda: screen(path, sys.argv[3]), "label": lambda: label(path, *sys.argv[3:5])}[stage]()
