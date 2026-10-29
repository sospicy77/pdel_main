"""노출값을 뺀 전체 변수 발생 모형 (03장 재분석). 끝난 단계는 건너뜀 (중단 후 재실행 가능).

노출값은 05장 취약성 틀의 노출 항이므로, 발생 모형에서는 빼고 나머지 변수(B_ne)만으로
1. run_selection2 (변수 선택 포함 nested CV) → 최고 −0.005 이내 최소 k 설정 선택 (run_v4 와 같은 규칙)
2. run_final (선택 변수 고정 CV, SHAP)
3. run_prespec all_ne (사전 지정 Firth 로지스틱, 8개 변수 + 안정성 선택)

사용법: python run_cls_ne.py <labeled 엑셀> <출력폴더>
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v4 import run  # noqa: E402

SET = "B_ne"


def main(lab, od):
    os.makedirs(od, exist_ok=True)
    run(f"{od}/log_s2.txt", "run_selection2.py", lab, "cls", SET, od)
    s2 = pd.read_csv(f"{od}/cls_{SET}_sel2_summary.csv")
    s2 = s2[s2.strategy != "boruta"].assign(k=lambda x: x.k.astype(int))
    ok = s2[s2.AUC_mean >= s2.AUC_mean.max() - 0.005].sort_values(["k", "AUC_mean"], ascending=[True, False])
    r = ok.iloc[0]
    cfg = {"set": SET, "strategy": r.strategy, "k": int(r.k), "model": r.model,
           "cv_nested": float(r.AUC_mean), "cv_nested_std": float(r.AUC_std), "best_any_k": float(s2.AUC_mean.max())}
    json.dump(cfg, open(f"{od}/config.json", "w"), ensure_ascii=False, indent=1)
    run(f"{od}/log_final.txt", "run_final.py", lab, "cls", SET, r.strategy, str(int(r.k)), r.model, od)
    run(f"{od}/log_prespec.txt", "run_prespec.py", lab, od, "all_ne")
    print("CLSNEDONE", cfg, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
