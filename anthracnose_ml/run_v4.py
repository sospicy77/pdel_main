"""v4: 피해도 임계값 기반 발생 분류 전체 실행 (중단되면 다시 실행하면 끝난 단계는 건너뜀).

1. run_threshold screen (임계값 0~0.03)
2. 전체 변수 세트(B)에서 AUC 가 가장 높은 임계값 선택 → 그 임계값으로 라벨을 바꾼 엑셀 생성
3. 세 변수세트마다 run_selection2 → 최종 설정 선택(최고 −0.005 이내 최소 k) → run_final → run_effects
     all        : B (기상 8개·IR 2개 포함 전체)
     noweather  : A (기상 정보는 노출값만)
     noweather_nt: A_nt (A 에서 강우-방제 타이밍 변수도 제외, 민감도 분석)

사용법: python run_v4.py <엑셀> <출력폴더>
"""
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"all": ("B", "all"), "noweather": ("A", "noweather"), "noweather_nt": ("A_nt", "noweather")}


def run(log, *args):
    if os.path.exists(log) and "DONE_OK" in open(log).read():
        return
    print(">>", " ".join(args), flush=True)
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(HERE, args[0]), *args[1:]], stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    with open(log, "a") as f:
        f.write("\nDONE_OK\n")


def main(xlsx, out):
    os.makedirs(out, exist_ok=True)
    run(f"{out}/log_screen.txt", "run_threshold.py", xlsx, "screen", out)
    s = pd.read_csv(f"{out}/thr_screen_summary.csv")
    sb = s[s.set == "B"]
    t = float(sb.loc[sb.AUC_mean.idxmax(), "threshold"])
    lab = f"{out}/labeled_t{t}.xlsx"
    if not os.path.exists(lab):
        subprocess.run([sys.executable, os.path.join(HERE, "run_threshold.py"), xlsx, "label", str(t), lab], check=True)
    cfg = {"threshold": t}
    for name, (sname, eff) in VARIANTS.items():
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
        run(f"{od}/log_effects.txt", "run_effects.py", lab, od, eff)
        json.dump(cfg, open(f"{out}/configs.json", "w"), ensure_ascii=False, indent=1)
    print("ALLDONE", cfg, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
