"""v4 보고서: 피해도 임계값으로 다시 정의한 발생 분류(전체 변수 / 기상 제외) + v3 피해도 회귀 섹션.

사용법: python make_report_v4.py <v4 엑셀> <results_v4> <v3 엑셀> <results_v3> <출력 html>
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from make_report import bar_chart, forest, line_chart, table  # noqa: E402
from make_report_v2 import FONTS, LEG_OR, LEG_RISK, imp_rows, or_table  # noqa: E402
from make_report_v3 import EXTRA_CSS, STYLE, build_fill  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT = {"meanrank": "평균순위", "rfshap": "RF-SHAP", "rfe": "RFE", "boruta": "Boruta"}
MODEL = {"ET_l3": "ExtraTrees", "RF_l1": "Random Forest", "RF_l3": "Random Forest", "RF_l5_m5": "Random Forest",
         "XGB": "XGBoost", "lin": "로지스틱(ElasticNet)"}
VLAB = {"all": "전체 변수 (기상 포함)", "noweather": "기상 제외 (노출값만)", "noweather_nt": "기상 제외 + 강우-방제 타이밍 제외"}


def main(x4, r4, x3, r3, out):
    fill = build_fill(x3, r3)          # 피해도 회귀 섹션 (v3 결과)
    cfg = json.load(open(f"{r4}/configs.json"))
    t = cfg["threshold"]
    d4 = C.load(x4)
    lab = pd.read_excel(f"{r4}/labeled_t{t}.xlsx", "모델링_데이터셋")
    npos = int(lab[C.Y1].sum())

    # 임계값 탐색
    s = pd.read_csv(f"{r4}/thr_screen_summary.csv")
    best = s.sort_values("AUC_mean", ascending=False).groupby(["threshold", "set"]).head(1)
    series = {}
    for sname, nm in [("B", "전체 변수"), ("A", "기상 제외 (노출값만)")]:
        g = best[best.set == sname].sort_values("threshold")
        series[nm] = [(f"{r.threshold:.3f}", r.AUC_mean) for r in g.itertuples()]
    rows = []
    for r in best[best.set == "B"].sort_values("threshold").itertuples():
        a = best[(best.set == "A") & (best.threshold == r.threshold)].iloc[0]
        rows.append([f"{r.threshold:.3f}", f"Y2 > {r.threshold:g}", r.n_pos, f"{r.prev * 100:.0f}%",
                     f"{r.AUC_mean:.3f} ± {r.AUC_std:.3f}", f"{r.PR_AUC_mean:.3f}", f"{r.BalAcc_mean:.3f}",
                     f"{a.AUC_mean:.3f}"])
    t_thr = table(pd.DataFrame(rows, columns=["임계값", "발생 정의", "발생 농가", "발생 비율", "AUC (전체 변수)",
                                              "PR-AUC", "균형정확도", "AUC (기상 제외)"]), "num")

    def variant(name):
        c = cfg[name]
        j = json.load(open(f"{r4}/{name}/cls_final.json"))
        ef = f"{r4}/{name}/effects.json" if name == "all" else f"{r4}/{name}/effects_noweather.json"
        e = json.load(open(ef))["cls"]
        return c, j, e

    V = {n: variant(n) for n in ["all", "noweather", "noweather_nt"]}
    t_var = table(pd.DataFrame([[VLAB[n], len(C.feature_sets(d4, stage2=False)[c["set"]]),
                                 f'{STRAT[c["strategy"]]} · {MODEL[c["model"]]} · k={c["k"]}',
                                 f'{c["cv_nested"]:.3f} ± {c["cv_nested_std"]:.3f}', f'{j["perf_fixed"]["AUC"]["mean"]:.3f}',
                                 f'{e["cv_auc"][0]:.3f}']
                                for n, (c, j, e) in V.items()],
                               columns=["모델", "후보 변수 수", "최종 설정", "AUC (선택 포함 CV)", "AUC (변수 고정)",
                                        "핵심변수 로지스틱 AUC"]), "num")

    def cls_fill(name, pre):
        c, j, e = V[name]
        return {f"{pre}_cfg": f'{STRAT[c["strategy"]]}로 고른 변수 {c["k"]}개 · {MODEL[c["model"]]}',
                f"{pre}_auc": f'{c["cv_nested"]:.3f}', f"{pre}_auc_fixed": f'{j["perf_fixed"]["AUC"]["mean"]:.3f}',
                f"{pre}_bar": bar_chart(imp_rows(j)[:15], "shap"), f"{pre}_forest": forest(e["table"]),
                f"{pre}_or": or_table(e["table"]), f"{pre}_core_auc": f'{e["cv_auc"][0]:.3f}',
                f"{pre}_pr2": f'{e["pseudoR2"]:.3f}'}

    fill.update(cls_fill("all", "ca"))
    fill.update(cls_fill("noweather", "cn"))
    fill.update(n=len(d4), n1=npos, n0=len(d4) - npos, thr=f"{t:g}", thr_pct=f"{t * 100:g}",
                thr_curve=line_chart(series, "AUC"), t_thr=t_thr, t_var=t_var,
                nt_auc=f'{cfg["noweather_nt"]["cv_nested"]:.3f}', leg_cls=LEG_RISK.format(w="발생 위험"),
                leg_or=LEG_OR)
    body = open(f"{HERE}/report_v4_body.html").read()
    for k, v in fill.items():
        body = body.replace("{{" + k + "}}", str(v))
    page = (f'<title>고추탄저병 최종 분석</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f"{FONTS}\n{STYLE}\n{EXTRA_CSS}\n{body}")
    open(out, "w").write(page)
    print("wrote", out, "unfilled:", re.findall(r"\{\{\w+\}\}", page))


if __name__ == "__main__":
    main(*sys.argv[1:6])
