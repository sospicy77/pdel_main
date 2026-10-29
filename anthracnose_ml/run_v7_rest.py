"""v7: 02·04·05장 재분석 (03장은 run_cls_ne, 06–08장은 run_reg_v4). 끝난 단계는 건너뜀.

1. run_threshold screen (임계값 0~0.03) — 라벨은 v4 와 같은 0.03 을 유지하고 결과만 비교
2. 기상 제외 모델 noweather(A)·noweather_nt(A_nt): run_selection2 → 설정 선택(run_v4 규칙) → run_final
   → run_effects noweather, noweather 는 run_prespec 까지
3. 취약성: run_adapt_compare → run_vulnerability
4. configs.json (all 은 run_cls_ne 결과를 그대로 기록)

사용법: python run_v7_rest.py <원자료 엑셀> <출력폴더>   (출력폴더/labeled_t0.03.xlsx, all_ne/ 가 있어야 함)
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v4 import run  # noqa: E402

T = 0.03


def main(xlsx, out):
    lab = f"{out}/labeled_t{T}.xlsx"
    run(f"{out}/log_screen.txt", "run_threshold.py", xlsx, "screen", out)
    cfg = {"threshold": T, "all": json.load(open(f"{out}/all_ne/config.json"))}
    for name, sname in [("noweather", "A"), ("noweather_nt", "A_nt")]:
        od = f"{out}/{name}"
        os.makedirs(od, exist_ok=True)
        run(f"{od}/log_s2.txt", "run_selection2.py", lab, "cls", sname, od)
        s2 = pd.read_csv(f"{od}/cls_{sname}_sel2_summary.csv")
        s2 = s2[s2.strategy != "boruta"].assign(k=lambda x: x.k.astype(int))
        ok = s2[s2.AUC_mean >= s2.AUC_mean.max() - 0.005].sort_values(["k", "AUC_mean"], ascending=[True, False])
        r = ok.iloc[0]
        cfg[name] = {"set": sname, "strategy": r.strategy, "k": int(r.k), "model": r.model,
                     "cv_nested": float(r.AUC_mean), "cv_nested_std": float(r.AUC_std),
                     "best_any_k": float(s2.AUC_mean.max())}
        run(f"{od}/log_final.txt", "run_final.py", lab, "cls", sname, r.strategy, str(int(r.k)), r.model, od)
        run(f"{od}/log_effects.txt", "run_effects.py", lab, od, "noweather")
        json.dump(cfg, open(f"{out}/configs.json", "w"), ensure_ascii=False, indent=1)
    run(f"{out}/noweather/log_prespec.txt", "run_prespec.py", lab, f"{out}/noweather", "noweather")
    vd = f"{out}/vulnerability"
    os.makedirs(vd, exist_ok=True)
    run(f"{out}/log_adapt.txt", "run_adapt_compare.py", lab, vd)
    run(f"{out}/log_vulnerability.txt", "run_vulnerability.py", lab, vd)
    print("V7RESTDONE", cfg, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
