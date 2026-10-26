"""전체 분석을 한 번에 다시 수행 (데이터 버전이 바뀌었을 때).

각 단계의 최적 설정은 v1 분석과 같은 규칙으로 자동 선택한다.
  - 변수세트: 1단계 요약에서 과제별 최고 성능 세트
  - 최종 설정: 2단계 요약에서 최고 성능 - 0.005 이내인 설정 중 변수 수가 가장 적은 것 (Boruta 제외)
  - 피해도 연속 회귀: 변환 비교 → 최고 변환에서 nested 튜닝·부분집합·ET 격자 → 격자 최고 설정으로 진단

사용법: python run_all.py <엑셀> <출력폴더>
"""
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRIC = {"cls": "AUC", "sev3": "rho", "reg": "R2"}


def run(*args, log):
    print(">>", " ".join(args), flush=True)
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(HERE, args[0]), *args[1:]],
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def pick_set(out, task):
    s = pd.read_csv(f"{out}/{task}_cv_summary.csv")
    return s.loc[s[f"{METRIC[task]}_mean"].idxmax(), "set"]


def pick_final(out, task, sname):
    s = pd.read_csv(f"{out}/{task}_{sname}_sel2_summary.csv")
    s = s[s.strategy != "boruta"].assign(k=lambda x: x.k.astype(int))
    m = f"{METRIC[task]}_mean"
    ok = s[s[m] >= s[m].max() - 0.005].sort_values(["k", m], ascending=[True, False])
    r = ok.iloc[0]
    return r.strategy, int(r.k), r.model, float(r[m])


def main(xlsx, out):
    os.makedirs(out, exist_ok=True)
    L = lambda name: f"{out}/log_{name}.txt"
    cfg = {}
    for t in ["cls", "reg", "sev3"]:
        run("run_selection.py", xlsx, t, out, log=L(f"s1_{t}"))
    for t in ["cls", "sev3", "reg"]:
        sname = pick_set(out, t)
        run("run_selection2.py", xlsx, t, sname, out, log=L(f"s2_{t}"))
        st, k, m, score = pick_final(out, t, sname)
        cfg[t] = {"set": sname, "strategy": st, "k": k, "model": m, "cv_nested": score}
        run("run_final.py", xlsx, t, sname, st, str(k), m, out, log=L(f"final_{t}"))
    json.dump(cfg, open(f"{out}/configs.json", "w"), ensure_ascii=False, indent=1)
    run("run_effects.py", xlsx, out, log=L("effects"))
    c1, c2 = cfg["cls"], cfg["reg"]
    run("run_hurdle.py", xlsx, c1["set"], c1["strategy"], str(c1["k"]), c1["model"],
        c2["strategy"], str(c2["k"]), c2["model"], out, log=L("hurdle"))

    # 피해도 연속 회귀 개선
    run("run_reg_tune.py", xlsx, "A", out, "A", log=L("regA"))
    ta = pd.read_csv(f"{out}/regtune_A_A.csv")
    best_t = ta.loc[ta.R2_t_mean.idxmax(), "transform"]
    cfg["reg_transform"] = best_t
    for st in ["B", "C"]:
        run("run_reg_tune.py", xlsx, st, out, best_t, "A", log=L(f"reg{st}"))
    run("run_reg_etgrid.py", xlsx, log=f"{out}/regtune_etgrid.txt")
    json.dump(cfg, open(f"{out}/configs.json", "w"), ensure_ascii=False, indent=1)
    print("ALLDONE", cfg, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
