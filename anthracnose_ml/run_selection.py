"""1단계: 변수세트 × 모델 × top-k 를 반복 K-fold 로 비교 (변수 선택은 fold 안에서 수행).

사용법: python run_selection.py <엑셀경로> <task: cls|reg|sev3> <출력폴더>
  cls  : Y1 발생라벨 (150건, 초발 관련 3변수 제외)
  reg  : 발병농가 67건의 logit(Y2) 회귀 (3변수 포함)
  sev3 : 발병농가 67건의 피해도 3단계(경/중/심) 분류 (3변수 포함)
"""
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

KS = [3, 5, 7, 10, 15, 20, 1000]   # 1000 = 전체 변수
N_SPLITS, N_REPEATS = 5, 6


def target(d, task):
    if task == "cls":
        return d, d[C.Y1].values, "cls"
    dd = d[d[C.Y1] == 1].reset_index(drop=True)
    if task == "reg":
        return dd, C.sev_to_logit(dd[C.Y2].values), "reg"
    # sev3: 순서형 3단계를 "심(≥20%) vs 나머지" 와 "중 이상 vs 경" 두 이진문제로 보지 않고,
    # 순위 점수(0/1/2)를 회귀로 예측한 뒤 Spearman 과 3단계 정확도로 평가
    return dd, pd.cut(dd[C.Y2], C.SEV_BINS, labels=False, right=False).values.astype(float), "reg"


def one_fold(X, y, tr, te, mtask, seed):
    rk = C.rank_features(X.iloc[tr], y[tr], mtask, seed)
    out = []
    models = C.CLS_MODELS if mtask == "cls" else C.REG_MODELS
    for k in KS:
        cols = list(rk.index[:k])
        for m in models:
            _, p = C.fit_predict(m, mtask, X.iloc[tr][cols], y[tr], X.iloc[te][cols], seed)
            out.append((k if k < 1000 else X.shape[1], m, te, p))
    return out, rk


def score(task, y, pred):
    if task == "cls":
        return {"AUC": roc_auc_score(y, pred)}
    if task == "reg":
        return {"R2": r2_score(y, pred), "rho": spearmanr(y, pred).correlation}
    cls = np.clip(np.rint(pred), 0, 2)
    return {"rho": spearmanr(y, pred).correlation, "acc3": (cls == y).mean()}


def main(path, task, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = C.load(path)
    dd, y, mtask = target(d, task)
    sets = C.feature_sets(dd, stage2=(task != "cls"))
    cv = (RepeatedStratifiedKFold if task == "cls" else RepeatedKFold)(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    yst = y if task == "cls" else np.zeros(len(y))
    rows, rank_freq = [], {}
    for sname, cols in sets.items():
        X = dd[cols]
        splits = list(cv.split(X, yst))
        res = Parallel(n_jobs=4)(delayed(one_fold)(X, y, tr, te, mtask, i)
                                 for i, (tr, te) in enumerate(splits))
        # 반복마다 out-of-fold 예측을 모아서 점수 계산
        preds = {}
        for i, (out, rk) in enumerate(res):
            rep = i // N_SPLITS
            for k, m, te, p in out:
                preds.setdefault((k, m, rep), np.full(len(y), np.nan))[te] = p
            for v in rk.index[:10]:
                rank_freq[(sname, v)] = rank_freq.get((sname, v), 0) + 1
        for (k, m, rep), p in preds.items():
            rows.append({"set": sname, "k": k, "model": m, "rep": rep, **score(task, y, p)})
        print(sname, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(f"{outdir}/{task}_cv_raw.csv", index=False)
    metric = {"cls": "AUC", "reg": "R2", "sev3": "rho"}[task]
    s = r.groupby(["set", "model", "k"]).agg(["mean", "std"]).drop(columns="rep")
    s.columns = ["_".join(c) for c in s.columns]
    s = s.sort_values(f"{metric}_mean", ascending=False)
    s.round(4).to_csv(f"{outdir}/{task}_cv_summary.csv")
    print(s.head(15).round(3).to_string())
    f = pd.Series(rank_freq).rename("top10_freq") / (N_SPLITS * N_REPEATS)
    f.round(3).sort_values(ascending=False).to_csv(f"{outdir}/{task}_top10_freq.csv")


if __name__ == "__main__":
    main(*sys.argv[1:4])
