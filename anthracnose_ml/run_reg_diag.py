"""피해도 회귀의 행 단위 진단: 어떤 농가(ID)가 교차검증 성능을 떨어뜨리는가.

- 반복 CV out-of-fold 예측으로 농가별 잔차와 오차제곱합(SSE) 기여율
- 해당 농가를 빼고 다시 CV 했을 때의 R² (제외 민감도)
- 같은 농가그룹(같은 기상·방제) 안 포장 간 피해도 차이

사용법: python run_reg_diag.py <엑셀> <변환> <set> <k> <model> <출력폴더>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from run_reg_tune import TRANSFORMS, base_models, data  # noqa: E402


def oof(X, z, k, model, n_rep=10, seed=0):
    cv = list(RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(X))

    def one(i, tr, te):
        cols = list(C.rank_features(X.iloc[tr], z[tr], "reg", i).index[:k])
        m = base_models(i)[model].fit(X.iloc[tr][cols], z[tr])
        return te, m.predict(X.iloc[te][cols])

    res = Parallel(n_jobs=4)(delayed(one)(i, tr, te) for i, (tr, te) in enumerate(cv))
    P = np.zeros((n_rep, len(z)))
    for i, (te, p) in enumerate(res):
        P[i // 5, te] = p
    return P


def main(path, tname, sname, k, model, outdir):
    k = int(k)
    dd, X, y = data(path, sname)
    z = TRANSFORMS[tname][0](y)
    P = oof(X, z, k, model)
    r2_rep = [r2_score(z, p) for p in P]
    pm = P.mean(0)
    res = z - pm
    t = pd.DataFrame({"ID": dd.ID, "농가그룹": dd["농가그룹"], "피해도": y, "발병주수": dd["발병주수"],
                      "예측_피해도": 10 ** pm if tname == "log10" else TRANSFORMS[tname][1](pm),
                      "잔차": res, "SSE기여율": res ** 2 / (res ** 2).sum()})
    t["방향"] = np.where(res > 0, "과소예측(실제가 더 심함)", "과대예측(실제가 더 가벼움)")
    g = dd.groupby("농가그룹")[C.Y2]
    t["같은농가_포장수"] = dd["농가그룹"].map(g.size())
    t["같은농가_피해도범위"] = dd["농가그룹"].map(g.agg(lambda v: f"{v.min():.3f}–{v.max():.3f}" if len(v) > 1 else "-"))
    t = t.sort_values("SSE기여율", ascending=False)

    # 상위 n 개 농가를 빼고 다시 CV (학습·평가 모두에서 제외)
    sens = []
    for n in [0, 2, 4, 6, 8, 10]:
        keep = ~dd.ID.isin(t.ID[:n])
        Pk = oof(X[keep].reset_index(drop=True), z[keep], k, model, n_rep=6, seed=1)
        sens.append({"제외_농가수": n, "제외_ID": ", ".join(t.ID[:n]),
                     "R2": float(np.mean([r2_score(z[keep], p) for p in Pk]))})
        print(sens[-1], flush=True)
    t.round(4).to_csv(f"{outdir}/regdiag_{tname}_{sname}_rows.csv", index=False)
    json.dump({"config": [tname, sname, k, model], "R2_mean": float(np.mean(r2_rep)),
               "R2_std": float(np.std(r2_rep)), "sensitivity": sens},
              open(f"{outdir}/regdiag_{tname}_{sname}.json", "w"), ensure_ascii=False, indent=1)
    print(t.head(15).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:7])
