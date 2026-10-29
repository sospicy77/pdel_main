"""적응력 요소 후보 조합을 같은 취약성 모형(run_vulnerability 의 곱셈형·로지스틱)으로 비교.

후보 조합
  S1 기존(안정성 선택)      : 강우 전 방제 경과일, 강우 후 방제 지연, 제초매트, 보호살균제 횟수
  S2 SHAP 상위 관행 4개     : 강우 전 방제 경과일, 7월 살포횟수, 강우 후 방제 지연, 살포일수
  S3 묶음 대표(살포일수)    : 방제 강도 = 살포일수, 강우 전 방제 경과일, 제초매트
  S4 묶음 대표(종합지수)    : 방제 강도 = 살포 변수 종합지수(1주성분), 강우 전 방제 경과일, 제초매트
  S5 S4 + 강우 후 방제 지연
  S6 S3 + 보호살균제 횟수
  S7–S9 방제강도 또는 타이밍 + 제초매트
  S10 보호살균제 횟수 + 강우 전 방제 경과일 + 제초매트, S11 보호살균제 횟수 + 제초매트 (v7 에서 추가)
평가: 반복 층화 5-fold CV AUC·로그손실 (곱셈형, 로지스틱), 전체 자료 로그우도·AIC, 부트스트랩 가중치 구간.

사용법: python run_adapt_compare.py <labeled 엑셀> <출력폴더>
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import run_vulnerability as V  # noqa: E402

warnings.filterwarnings("ignore")
INTENSITY = ["7월_살포횟수", "살포일수", "보호_카_횟수", "6월_살포횟수", "5월_살포횟수", "8월_살포횟수"]
SETS = {
    "S1 기존(안정성 선택)": [("7~8월_사전경과_최대", -1), ("7~8월_사후지연_최대", -1), ("제초매트", 1), ("보호_카_횟수", 1)],
    "S2 SHAP 상위 관행 4개": [("7~8월_사전경과_최대", -1), ("7월_살포횟수", 1), ("7~8월_사후지연_최대", -1), ("살포일수", 1)],
    "S3 방제강도(살포일수)+타이밍+제초매트": [("살포일수", 1), ("7~8월_사전경과_최대", -1), ("제초매트", 1)],
    "S4 방제강도(종합지수)+타이밍+제초매트": [("방제강도_지수", 1), ("7~8월_사전경과_최대", -1), ("제초매트", 1)],
    "S5 S4+강우 후 방제 지연": [("방제강도_지수", 1), ("7~8월_사전경과_최대", -1), ("제초매트", 1), ("7~8월_사후지연_최대", -1)],
    "S6 S3+보호살균제": [("살포일수", 1), ("7~8월_사전경과_최대", -1), ("제초매트", 1), ("보호_카_횟수", 1)],
    "S7 방제강도(종합지수)+제초매트": [("방제강도_지수", 1), ("제초매트", 1)],
    "S8 살포일수+제초매트": [("살포일수", 1), ("제초매트", 1)],
    "S9 강우 전 방제 경과일+제초매트": [("7~8월_사전경과_최대", -1), ("제초매트", 1)],
    # v7(보호살균제 횟수 수정)에서 이 변수가 발생 모델 1순위가 되어 추가한 후보
    "S10 보호살균제+타이밍+제초매트": [("보호_카_횟수", 1), ("7~8월_사전경과_최대", -1), ("제초매트", 1)],
    "S11 보호살균제+제초매트": [("보호_카_횟수", 1), ("제초매트", 1)],
}


def intensity_index(d):
    Z = d[INTENSITY].fillna(d[INTENSITY].median())
    Z = (Z - Z.mean()) / Z.std()
    pca = PCA(1).fit(Z)
    s = pca.transform(Z)[:, 0]
    if np.corrcoef(s, Z["살포일수"])[0, 1] < 0:          # 클수록 방제를 많이 한 쪽이 되도록
        s, comp = -s, -pca.components_[0]
    else:
        comp = pca.components_[0]
    return s, dict(zip(INTENSITY, np.round(comp, 3))), float(pca.explained_variance_ratio_[0])


def boot_w(b, y, E, S, size, a):
    rng = np.random.default_rng(500 + b)
    i = rng.integers(0, len(y), len(y))
    return V.Mult().fit(y[i], E[i], S[i], size[i], a[i], seed=b).params()["w"]


def main(xlsx, out):
    os.makedirs(out, exist_ok=True)
    d = C.load(xlsx)
    d["방제강도_지수"], loadings, evr = intensity_index(d)
    y = d[C.Y1].values.astype(float)
    res = {"intensity_loadings": loadings, "intensity_evr": evr}
    # 관행 목록을 모듈 전역으로 바꿔 쓰므로, 작업자가 같은 전역을 보도록 스레드 병렬 사용
    for name, prs in SETS.items():
        V.PRACTICES = prs
        V.P_NAMES = [p for p, _ in prs]
        E, S, size, a, ref = V.prepare(d)
        m = V.Mult().fit(y, E, S, size, a)
        with parallel_config(backend="threading", n_jobs=4):
            cv = V.cv_compare(y, E, S, size, a, pairs=[], n_rep=10)
            boots = Parallel()(delayed(boot_w)(b, y, E, S, size, a) for b in range(200))
        w_ci = {p: [float(np.percentile([bt[p] for bt in boots], q)) for q in (2.5, 97.5)] for p in V.P_NAMES}
        p = m.params()
        res[name] = {"practices": V.P_NAMES, "loglik": float(m.loglik), "k": int(m.k),
                     "AIC": float(2 * m.k - 2 * m.loglik), "cv": cv, "w": p["w"], "w_ci": w_ci,
                     "theta_x": p["theta_x"], "alpha": p["alpha"]}
        print(name, "| CV AUC mult", round(cv["B 곱셈형"]["AUC"][0], 3), "logit", round(cv["A 로지스틱+E×관행"]["AUC"][0], 3),
              "| logloss mult", round(cv["B 곱셈형"]["logloss"], 3), "| AIC", round(res[name]["AIC"], 1),
              "| w", {k: round(v, 2) for k, v in p["w"].items()},
              "| w CI", {k: [round(x, 2) for x in v] for k, v in w_ci.items()}, flush=True)
    print("intensity loadings", loadings, "EVR", round(evr, 3))
    json.dump(res, open(f"{out}/adapt_compare.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:3])
