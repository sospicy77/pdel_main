"""7월 살포횟수가 적응력 요소로 추가 선정되려면 어떤 농가의 값이 바뀌어야 하는가 (재조사 대상 선정용).

모형: run_vulnerability 의 곱셈형 취약성 모형. 적응력 = 현재 2개(강우 전 방제 경과일, 제초매트) + 7월 살포횟수.
"선정" 기준 (run_adapt_compare 에서 S9 를 고를 때와 같은 기준)
  1) 7월 살포횟수 가중치의 부트스트랩 95% 구간이 0 을 벗어남
  2) AIC 가 현재 모형(S9, 2개)보다 작음
  3) 반복 CV AUC(곱셈형)가 현재 모형보다 큼
절차
  0단계: 각 농가의 7월 살포횟수만 "모형 방향"(살포가 많을수록 위험↓) 쪽으로 바꿔 봄
         (발생 농가는 현재값보다 작게, 미발생 농가는 크게, 0~8 범위) → 농가별 최대 AIC 개선량으로 순위
  탐욕적 수정: 매 단계 AIC 를 가장 많이 낮추는 농가를 고르되, 그 농가 최대 개선량의 80% 이상을 내는
         가장 작은 변경값을 채택 → 기준 2) 를 만족하면 1)·3) 을 확인, 모두 만족할 때까지 반복
결과는 "이 값이 틀렸다"는 뜻이 아니라 모형 방향과 가장 크게 어긋나는 응답이라는 뜻이며, 재조사로 확인된 값만 반영해야 한다.

사용법: python run_july_influence.py <labeled 엑셀> <출력폴더>
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import run_vulnerability as V  # noqa: E402

warnings.filterwarnings("ignore")
JUL = "7월_살포횟수"
BASE = [("7~8월_사전경과_최대", -1), ("제초매트", +1)]
PLUS = BASE + [(JUL, +1)]
MAX_STEPS, KEEP = 15, 0.8


def fit(d, prs, y, restarts_seed=0):
    V.PRACTICES, V.P_NAMES = prs, [p for p, _ in prs]
    E, S, size, a, _ = V.prepare(d)
    m = V.Mult().fit(y, E, S, size, a, seed=restarts_seed)
    return m, (E, S, size, a)


def aic_with(d, y, i, v):
    dd = d.copy()
    dd.loc[i, JUL] = v
    m, _ = fit(dd, PLUS, y)
    return float(2 * m.k - 2 * m.loglik), float(m.params()["w"][JUL])


def candidates(d, y, i):
    x = int(d.loc[i, JUL])
    return list(range(0, x)) if y[i] == 1 else list(range(x + 1, 9))


def scan(d, y, rows):
    jobs = [(i, v) for i in rows for v in candidates(d, y, i)]
    out = Parallel(n_jobs=4)(delayed(aic_with)(d, y, i, v) for i, v in jobs)
    res = pd.DataFrame([(i, v, a, w) for (i, v), (a, w) in zip(jobs, out)], columns=["row", "value", "AIC", "w"])
    return res


def check(d, y):
    """기준 1)·3): 부트스트랩 가중치 구간과 반복 CV AUC."""
    from joblib import parallel_config
    V.PRACTICES, V.P_NAMES = PLUS, [p for p, _ in PLUS]
    E, S, size, a, _ = V.prepare(d)

    def boot(b):
        rng = np.random.default_rng(500 + b)
        i = rng.integers(0, len(y), len(y))
        return V.Mult().fit(y[i], E[i], S[i], size[i], a[i], seed=b).params()["w"][JUL]
    with parallel_config(backend="threading", n_jobs=4):
        ws = Parallel()(delayed(boot)(b) for b in range(200))
        cv = V.cv_compare(y, E, S, size, a, pairs=[], n_rep=10)
    return [float(np.percentile(ws, 2.5)), float(np.percentile(ws, 97.5))], cv["B 곱셈형"]["AUC"]


def main(xlsx, out):
    os.makedirs(out, exist_ok=True)
    d = C.load(xlsx)
    y = d[C.Y1].values.astype(float)
    m0, (E, S, size, a) = fit(d, BASE, y)
    aic0 = float(2 * m0.k - 2 * m0.loglik)
    from joblib import parallel_config
    with parallel_config(backend="threading", n_jobs=4):
        cv0 = V.cv_compare(y, E, S, size, a, pairs=[], n_rep=10)["B 곱셈형"]["AUC"]
    mp, _ = fit(d, PLUS, y)
    res = {"base": {"AIC": aic0, "cv_auc": cv0, "w": m0.params()["w"]},
           "plus_original": {"AIC": float(2 * mp.k - 2 * mp.loglik), "w": mp.params()["w"]}}
    print("S9 AIC", round(aic0, 2), "CV AUC", cv0, "| +July AIC", round(res["plus_original"]["AIC"], 2), mp.params()["w"], flush=True)

    # 0단계: 농가별 단독 변경 효과
    s0 = scan(d, y, range(len(d)))
    best0 = s0.loc[s0.groupby("row").AIC.idxmin()].sort_values("AIC")
    best0["ID"] = d.ID.values[best0.row]
    best0["원래값"] = d[JUL].values[best0.row]
    best0["발생"] = y[best0.row].astype(int)
    best0["AIC 개선"] = res["plus_original"]["AIC"] - best0.AIC
    best0.to_csv(f"{out}/july_single_scan.csv", index=False)
    print(best0.head(20)[["ID", "발생", "원래값", "value", "AIC 개선", "w"]].to_string(), flush=True)

    # 탐욕적 수정
    cur, steps, changed = d.copy(), [], []
    for step in range(MAX_STEPS):
        rows = [i for i in range(len(d)) if i not in changed and candidates(cur, y, i)]
        s = scan(cur, y, rows)
        g = s.groupby("row")
        mc = fit(cur, PLUS, y)[0]
        base_aic = 2 * mc.k - 2 * mc.loglik
        gain = (base_aic - g.AIC.min()).sort_values(ascending=False)
        i = int(gain.index[0])
        si = s[s.row == i].assign(gain=lambda t: base_aic - t.AIC)
        x0 = int(cur.loc[i, JUL])
        ok = si[si.gain >= KEEP * si.gain.max()].assign(dist=lambda t: (t.value - x0).abs()).sort_values("dist")
        r = ok.iloc[0]
        cur.loc[i, JUL] = int(r.value)
        changed.append(i)
        rec = {"step": step + 1, "ID": d.ID[i], "occurrence": int(y[i]), "from": x0, "to": int(r.value),
               "AIC": float(r.AIC), "w_july": float(r.w),
               "days_before_rain": float(d.loc[i, "7~8월_사전경과_최대"]), "weed_mat": int(d.loc[i, "제초매트"]),
               "exposure": float(d.loc[i, "노출값"]), "disease_last_year": int(d.loc[i, "작년_탄저발생"])}
        print("step", rec, flush=True)
        if rec["AIC"] < aic0 and rec["w_july"] > 0:
            ci, cv = check(cur, y)
            rec.update(w_ci=ci, cv_auc=cv)
            print("   check: w CI", [round(c, 3) for c in ci], "CV AUC", [round(c, 3) for c in cv], flush=True)
            steps.append(rec)
            if ci[0] > 0 and cv[0] > cv0[0]:
                break
        else:
            steps.append(rec)
    res["steps"] = steps
    m, _ = fit(cur, PLUS, y)
    res["final"] = {"AIC": float(2 * m.k - 2 * m.loglik), "w": m.params()["w"], "n_changed": len(changed)}
    json.dump(res, open(f"{out}/july_influence.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("JULYDONE", res["final"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
