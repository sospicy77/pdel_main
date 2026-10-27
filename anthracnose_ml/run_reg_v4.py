"""v4 데이터로 피해도 회귀(Y2, 발병농가) 전체 재실행. 끝난 단계는 건너뜀 (중단 후 재실행 가능).

1. run_reg_final compare  → 15개 모델 × 척도 × 변수세트 × k
2. run_reg_final nested   → 선택 편향 없는 성능
3. final: 계열별(선형·커널·트리) 최고 설정 + ExtraTrees k=7 (SHAP 해석용)

사용법: python run_reg_v4.py <엑셀> <출력폴더>
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_reg_final as R  # noqa: E402

LINEAR, KERNEL, TREE = ["Ridge", "EN", "Huber"], ["KRR", "GPR", "SVR", "SVR_b"], ["ET", "ET_b", "RF"]


def main(xlsx, out):
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/reg3_compare.csv"):
        R.compare(xlsx, out)
    if not os.path.exists(f"{out}/reg3_nested.json"):
        R.nested(xlsx, out)
    s = pd.read_csv(f"{out}/reg3_compare.csv")
    s = s[s.k > 0]
    scale, sname = s.iloc[0][["scale", "set"]]
    g = s[(s.scale == scale) & (s.set == sname)].sort_values("R2_log_mean", ascending=False)
    picks = {}
    for tag, fam in [("ridge", LINEAR), ("krr", KERNEL), ("et", TREE)]:
        r = g[g.model.isin(fam)].iloc[0]
        picks[tag] = (r.model, int(r.k))
    et7 = g[g.model.isin(TREE) & (g.k == 7)].iloc[0]
    picks["et7"] = (et7.model, 7)
    json.dump({"scale": scale, "set": sname, "picks": picks}, open(f"{out}/reg_picks.json", "w"),
              ensure_ascii=False, indent=1)
    for tag, (m, k) in picks.items():
        if not os.path.exists(f"{out}/reg3_{tag}.json"):
            R.final(xlsx, out, scale, sname, k, m, tag)
    print("REGV4DONE", picks, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
