"""최종 분석(v3) 보고서: 발생 분류(Y1) + 피해도 회귀(Y2, 15개 모델 비교).

사용법: python make_report_v3.py <v3 엑셀> <results_v3> <출력 html>
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from make_report import E, bar_chart, dependence, forest, line_chart, table  # noqa: E402
from make_report_v2 import FONTS, LEG_OR, LEG_RISK, imp_rows, or_table  # noqa: E402
from run_reg_final import FAMILY  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = re.search(r"<style>.*?</style>", open(f"{HERE}/report_template.html").read(), re.S).group(0)
EXTRA_CSS = """<style>
:root { --fam1: #2a78d6; --fam2: #1baf7a; --fam3: #eb6834; --fam4: #c98500; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --fam1: #3987e5; --fam2: #199e70; --fam3: #d95926; --fam4: #eda100; } }
:root[data-theme="dark"] { --fam1: #3987e5; --fam2: #199e70; --fam3: #d95926; --fam4: #eda100; }
.bar.f1 { fill: var(--fam1); } .bar.f2 { fill: var(--fam2); } .bar.f3 { fill: var(--fam3); } .bar.f4 { fill: var(--fam4); }
</style>"""
FAMCLS = {"선형": "f1", "커널": "f2", "거리": "f2", "트리": "f3", "부스팅": "f4", "앙상블": "f1"}
MODEL_LAB = {"EN": "ElasticNet", "Ridge": "Ridge", "Huber": "Huber", "SVR": "SVR (기본)", "SVR_b": "SVR (튜닝)",
             "KRR": "Kernel Ridge", "GPR": "가우시안 프로세스", "kNN": "kNN", "RF": "Random Forest",
             "ET": "ExtraTrees (기본)", "ET_b": "ExtraTrees (튜닝)", "GBR": "Gradient Boosting",
             "XGB": "XGBoost", "LGBM": "LightGBM", "CatBoost": "CatBoost"}


def model_bar(rows, width=660, label_w=200):
    """모델별 최고 R² (log10). rows: dict(label, v, fam, k)."""
    bh, gap = 20, 6
    h = len(rows) * (bh + gap) + 28
    vmax = 0.6
    plot_w = width - label_w - 80
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    for t in np.arange(0, vmax + 1e-9, 0.1):
        x = label_w + t / vmax * plot_w
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{h-24}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h-8}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(rows):
        y = i * (bh + gap)
        w = max(2, r["v"] / vmax * plot_w)
        tip = f'{r["label"]}: R² {r["v"]:.3f} (k={r["k"]}, {r["fam"]})'
        out.append(f'<g class="mark"><title>{E(tip)}</title>'
                   f'<rect x="0" y="{y}" width="{width}" height="{bh+gap}" class="hit"/>'
                   f'<text x="{label_w-8}" y="{y+bh/2+4}" class="lab" text-anchor="end">{E(r["label"])}</text>'
                   f'<path d="M{label_w},{y} h{w-4:.1f} q4,0 4,4 v{bh-8} q0,4 -4,4 h-{w-4:.1f} z" class="bar {FAMCLS[r["fam"]]}"/>'
                   f'<text x="{label_w+w+6:.1f}" y="{y+bh/2+4}" class="val">{r["v"]:.3f}</text></g>')
    out.append("</svg>")
    return "".join(out)


def build_fill(xlsx, rd):
    d = C.load(xlsx)
    n1 = int(d[C.Y1].sum())
    cls = json.load(open(f"{rd}/cls_final.json"))
    eff = json.load(open(f"{rd}/effects.json"))
    s2 = pd.read_csv(f"{rd}/cls_B_sel2_summary.csv")
    cls_nested = s2[(s2.strategy == "rfe") & (s2.model == "ET_l3") & (s2.k.astype(str) == "25")].iloc[0]

    cmp_ = pd.read_csv(f"{rd}/reg3_compare.csv")
    nested = json.load(open(f"{rd}/reg3_nested.json"))
    R = {t: json.load(open(f"{rd}/reg3_{t}.json")) for t in ["ridge", "krr", "et", "et7"]}

    lg = cmp_[(cmp_.scale == "log10") & (cmp_.set == "A")]
    best = lg.sort_values("R2_log_mean", ascending=False).drop_duplicates("model")
    rows = [{"label": MODEL_LAB.get(r.model, r.model), "v": r.R2_log_mean,
             "fam": "앙상블" if r.k == -1 else FAMILY[r.model], "k": "-" if r.k == -1 else int(r.k)}
            for r in best.itertuples()]
    t_models = table(pd.DataFrame([[i + 1, MODEL_LAB.get(r.model, r.model),
                                    "앙상블" if r.k == -1 else FAMILY[r.model],
                                    r.members if r.k == -1 else int(r.k),
                                    f"{r.R2_log_mean:.3f} ± {r.R2_log_std:.3f}", f"{r.R2_raw_mean:.3f}",
                                    f"{r.rho_mean:.3f}", f"{r.RMSE_raw_mean:.3f}"]
                                   for i, r in enumerate(best.itertuples())],
                                  columns=["순위", "모델", "계열", "변수 수 k", "R² (log10 Y2)", "R² (Y2 원척도)",
                                           "Spearman ρ", "RMSE (Y2)"]), "num")

    # 척도·세트 비교
    lab = lambda r: r.model if r.k == -1 else f"{MODEL_LAB.get(r.model, r.model)} (k={r.k})"
    sc_rows = []
    for (scale, sname), g in cmp_.groupby(["scale", "set"]):
        b1 = g.sort_values("R2_log_mean", ascending=False).iloc[0]
        b2 = g.sort_values("R2_raw_mean", ascending=False).iloc[0]
        sc_rows.append([{"log10": "log10(Y2)", "raw": "Y2 원값"}[scale],
                        {"A": "A: 기상 제외", "B": "B: 기상 포함"}[sname],
                        lab(b1), f"{b1.R2_log_mean:.3f}", lab(b2), f"{b2.R2_raw_mean:.3f}"])
    t_scale = table(pd.DataFrame(sc_rows, columns=["반응 척도", "변수세트", "log10 R² 최고 모델", "R² (log10)",
                                                   "원척도 R² 최고 모델", "R² (원척도)"]), "num")

    # k 곡선
    series = {}
    for m, lab in [("Ridge", "Ridge"), ("KRR", "Kernel Ridge"), ("ET_b", "ExtraTrees")]:
        g = lg[lg.model == m].sort_values("k")
        series[lab] = [(int(r.k), r.R2_log_mean) for r in g.itertuples()]

    # 최종 모델 성능 표
    t_final = table(pd.DataFrame([[lab, ", ".join(R[t]["selected"]),
                                   f'{R[t]["cv"]["R2_log"][0]:.3f} ± {R[t]["cv"]["R2_log"][1]:.3f}',
                                   f'{R[t]["cv"]["R2_raw"][0]:.3f}', f'{R[t]["cv"]["rho"][0]:.3f}']
                                  for t, lab in [("ridge", "Ridge (선형)"), ("krr", "Kernel Ridge (비선형 커널)"),
                                                 ("et", "ExtraTrees (트리, k=3)"), ("et7", "ExtraTrees (트리, k=7)")]],
                                 columns=["모델", "선택 변수", "R² (log10)", "R² (원척도)", "Spearman ρ"]), "num")

    # Ridge 계수
    dd = d[d[C.Y1] == 1].reset_index(drop=True)
    coef = R["ridge"]["coef_std"]
    t_coef = table(pd.DataFrame([[v, "0→1" if dd[v].nunique() == 2 else f"+1 SD ({dd[v].std():.1f})",
                                  f"{c:+.3f}", f"×{10 ** c:.2f}"] for v, c in coef.items()],
                                columns=["변수", "단위", "log10 계수 (표준화)", "피해도 배수"]), "num")

    def perm_table(j):
        return table(pd.DataFrame([[r["var"], f'{r["perm_mean"]:+.3f} ± {r["perm_std"]:.3f}',
                                    f'{r["fold_sel"] * 100:.0f}%'] for r in j["importance"] if r["fold_sel"] >= 0.1],
                                  columns=["변수", "Permutation ΔR²", "fold 선택률"]), "num")

    et = R["et7"]
    et_rows = [{"var": r["var"], "shap": r["mean_abs_shap"], "rho": r["direction"]} for r in et["importance"]
               if r["var"] in et["selected"]]
    et_rows.sort(key=lambda r: -r["shap"])
    sv = pd.read_csv(f"{rd}/reg3_et7_shap.csv")
    Xs = pd.read_csv(f"{rd}/reg3_et7_X.csv")
    deps = "".join(f'<figure class="dep">{dependence(Xs[v], sv[v], v)}</figure>' for v in [r["var"] for r in et_rows[:6]])

    rr = pd.read_csv(f"{rd}/reg3_ridge_rows.csv").head(12)
    t_rows = table(pd.DataFrame([[i + 1, r.ID, f"{r.피해도:.4f}", f"{r.예측_피해도:.3f}", f"{r.잔차:+.2f}",
                                  f"{r.SSE기여율 * 100:.1f}%", "과소예측" if r.잔차 > 0 else "과대예측",
                                  int(r.이환과_즉시제거), int(r.초발_DOY)]
                                 for i, r in enumerate(rr.itertuples())],
                                columns=["순위", "ID", "실제 피해도", "예측 피해도", "잔차 (log10)", "SSE 기여", "유형",
                                         "이환과 즉시제거", "초발 DOY"]), "num")
    t_sens = table(pd.DataFrame([[s["n"], f'{s["R2_log"]:.3f}', ", ".join(s["ids"]) or "-"]
                                 for s in R["ridge"]["sensitivity"]],
                                columns=["제외 농가 수", "R² (log10)", "제외한 ID"]), "num")

    fill = dict(
        n=len(d), n1=n1, n0=len(d) - n1,
        cls_auc=f"{cls_nested.AUC_mean:.3f}", cls_auc_fixed=f'{cls["perf_fixed"]["AUC"]["mean"]:.3f}',
        reg_r2=f'{R["ridge"]["cv"]["R2_log"][0]:.3f}', reg_nested=f'{nested["nested"]["R2_log"][0]:.3f}',
        krr_r2=f'{R["krr"]["cv"]["R2_log"][0]:.3f}', et_r2=f'{R["et"]["cv"]["R2_log"][0]:.3f}',
        cls_bar=bar_chart(imp_rows(cls)[:15], "shap"), leg_cls=LEG_RISK.format(w="발생 위험"),
        cls_forest=forest(eff["cls"]["table"]), cls_or=or_table(eff["cls"]["table"]), leg_or=LEG_OR,
        cls_core_auc=f'{eff["cls"]["cv_auc"][0]:.3f}',
        model_bar=model_bar(rows), t_models=t_models, t_scale=t_scale,
        curve=line_chart(series, "R²"),
        nested_r2=f'{nested["nested"]["R2_log"][0]:.3f} ± {nested["nested"]["R2_log"][1]:.3f}',
        nested_freq=", ".join(f"{k} {v}회" for k, v in nested["chosen_freq"].items()),
        t_final=t_final, t_coef=t_coef,
        krr_perm=perm_table(R["krr"]),
        et_bar=bar_chart(et_rows, "shap"), leg_sev=LEG_RISK.format(w="피해"), deps=deps,
        t_rows=t_rows, t_sens=t_sens,
    )
    return fill


def main(xlsx, rd, out):
    fill = build_fill(xlsx, rd)
    body = open(f"{HERE}/report_v3_body.html").read()
    for k, v in fill.items():
        body = body.replace("{{" + k + "}}", str(v))
    page = (f'<title>고추탄저병 최종 분석</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f"{FONTS}\n{STYLE}\n{EXTRA_CSS}\n{body}")
    open(out, "w").write(page)
    print("wrote", out, "unfilled:", re.findall(r"\{\{\w+\}\}", page))


if __name__ == "__main__":
    main(*sys.argv[1:4])
