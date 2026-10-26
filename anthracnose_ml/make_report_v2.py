"""수정 데이터(v2) 재분석 보고서 + v1 대비 비교.

사용법: python make_report_v2.py <v2 엑셀> <v1 엑셀> <results_v2> <출력 html>
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from make_report import E, bar_chart, forest, table  # noqa: E402
from run_reg_tune import data  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = re.search(r"<style>.*?</style>", open(f"{HERE}/report_template.html").read(), re.S).group(0)
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="stylesheet" '
         'href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap">')
LEG_RISK = ('<div class="legend"><span><i style="background:var(--up)"></i>값이 클수록 {w}↑</span>'
            '<span><i style="background:var(--down)"></i>값이 클수록 {w}↓</span>'
            '<span><i style="background:var(--flat)"></i>비단조</span></div>')
LEG_OR = ('<div class="legend"><span><i style="background:var(--up)"></i>95% CI &gt; 1</span>'
          '<span><i style="background:var(--down)"></i>95% CI &lt; 1</span>'
          '<span><i style="background:var(--flat)"></i>CI가 1을 포함</span></div>')


def imp_rows(j):
    return [{"var": r["var"], "shap": r["mean_abs_shap"], "rho": r["direction_rho"]} for r in j["importance"]]


def or_table(rows):
    return table(pd.DataFrame([[r["var"], r["unit"], f'{r["OR"]:.2f}', f'{r["lo"]:.2f}–{r["hi"]:.2f}', f'{r["p"]:.3f}']
                               for r in rows], columns=["변수", "단위", "OR", "95% CI", "p"]), "num")


def main(xlsx, v1_xlsx, r2, out):
    cfg = json.load(open(f"{r2}/configs.json"))
    J = {t: json.load(open(f"{r2}/{t}_final.json")) for t in ["cls", "sev3"]}
    eff = json.load(open(f"{r2}/effects.json"))
    d = C.load(xlsx)
    n1 = int(d[C.Y1].sum())

    # 버전 비교 표
    cmp_ = pd.read_csv(f"{r2}/compare_versions.csv")
    cmp_["v"] = cmp_["mean"].map("{:.3f}".format) + " ± " + cmp_["std"].map("{:.3f}".format)
    metric = {"cls": "AUC", "sev3": "Spearman ρ", "logit": "R² (logit)", "log10": "R² (log10)"}
    task_lab = {"cls": "발생 분류", "sev3": "피해도 3단계", "logit": "피해도 회귀", "log10": "피해도 회귀"}
    pv = cmp_.pivot_table(index=["과제", "파이프라인"], columns="데이터", values="v", aggfunc="first", sort=False)
    pv = pv[["v1 (150건)", "삭제만 반영 (142건)", "v2 (142건)"]].reset_index()
    pv.insert(1, "지표", pv["과제"].map(metric))
    pv["과제"] = pv["과제"].map(task_lab)
    t_cmp = table(pv.rename(columns={"v1 (150건)": "v1 (150건)", "삭제만 반영 (142건)": "삭제만 (142건)"}), "num")

    # 데이터 변경 내역
    meta = json.load(open(f"{r2}/compare_meta.json"))
    v1 = C.load(v1_xlsx)
    gone = v1[v1.ID.isin(meta["deleted"])][["ID", C.Y1, C.Y2, "발병주수"]]
    t_del = table(pd.DataFrame([[r.ID, int(r[C.Y1]), f"{r[C.Y2]:.4f}", int(r['발병주수'])] for _, r in gone.iterrows()],
                               columns=["삭제된 ID", "발생", "피해도", "발병주수"]), "num")

    # 피해도 회귀: 변환 비교, 튜닝, 최종 계수
    ta = pd.read_csv(f"{r2}/regtune_A_A.csv")
    tlab = {"logit": "logit", "log10": "log10", "sqrt": "제곱근", "asin": "arcsine √", "raw": "원값"}
    t_tr = table(pd.DataFrame([[tlab[t], g.iloc[0].model, g.iloc[0].k, f'{g.iloc[0].R2_t_mean:.3f}',
                                f'{g.iloc[0].R2_raw_mean:.3f}', f'{g.iloc[0].rho_mean:.3f}']
                               for t, g in ta.groupby("transform", sort=False)],
                              columns=["변환", "최고 모델", "k", "R² (변환 척도)", "R² (원척도)", "Spearman ρ"]), "num")
    tb = pd.read_csv(f"{r2}/regtune_B_log10_A.csv", index_col=0)
    tc = pd.read_csv(f"{r2}/regtune_C_log10_A.csv", index_col=0)
    grid = sorted([l.rsplit(") ", 1) for l in open(f"{r2}/regtune_etgrid.txt").read().splitlines()],
                  key=lambda g: -float(g[1].split()[0]))
    best_en = ta[ta["transform"] == "log10"].iloc[0]
    rows = [["변환 비교 단계 최고 (평균순위 k, 기본 하이퍼파라미터)", f"{best_en.model} k={best_en.k}",
             f"{best_en.R2_t_mean:.3f} ± {best_en.R2_t_std:.3f}"]]
    rows += [["nested 튜닝 (k·하이퍼파라미터 모두 안쪽 CV)", m, f'{tb.loc[m, "R2_t_mean"]:.3f} ± {tb.loc[m, "R2_t_std"]:.3f}']
             for m in tb.sort_values("R2_t_mean", ascending=False).index]
    rows += [["최적 부분집합 (PRESS 최소)", m, f'{tc.loc[m, "R2_t_mean"]:.3f} ± {tc.loc[m, "R2_t_std"]:.3f}'] for m in tc.index]
    rows += [["ExtraTrees 격자 탐색", f"(k, 잎, max_features) = {g[0]})", g[1]] for g in grid[:2]]
    t_tune = table(pd.DataFrame(rows, columns=["방식", "모델 / 설정", "R² (log10)"]), "num")

    dd, X, y = data(xlsx, "A")
    z = np.log10(y)
    cols = list(C.rank_features(X, z, "reg", 0).index[:3])
    Z = X[cols].fillna(X[cols].median())
    Z = (Z - Z.mean()) / Z.std()
    ols = sm.OLS(z, sm.add_constant(Z)).fit()
    ci = ols.conf_int()
    t_coef = table(pd.DataFrame([[c, "0→1" if X[c].nunique() == 2 else f"+1 SD ({X[c].std():.1f})",
                                  f"{ols.params[c]:+.3f}", f"×{10 ** ols.params[c]:.2f}",
                                  f"{ci.loc[c, 0]:+.3f} – {ci.loc[c, 1]:+.3f}", f"{ols.pvalues[c]:.3f}"] for c in cols],
                                columns=["변수", "단위", "log10 계수", "감염주율 배수", "95% CI", "p"]), "num")

    # 문제 농가
    diag = json.load(open(f"{r2}/regdiag_log10_A.json"))
    rd = pd.read_csv(f"{r2}/regdiag_log10_A_rows.csv").head(12)
    t_rows = table(pd.DataFrame([[i + 1, r.ID, f"{r.피해도:.4f}", int(r.발병주수), f"{r.예측_피해도:.3f}",
                                  f"{r.잔차:+.2f}", f"{r.SSE기여율 * 100:.1f}%", r.방향.split("(")[0]]
                                 for i, r in enumerate(rd.itertuples())],
                                columns=["순위", "ID", "실제 피해도", "발병주수", "예측 피해도", "잔차 (log10)",
                                         "SSE 기여", "유형"]), "num")
    t_sens = table(pd.DataFrame([[s["제외_농가수"], f'{s["R2"]:.3f}', s["제외_ID"] or "-"] for s in diag["sensitivity"]],
                                columns=["제외 농가 수", "R² (log10)", "제외한 ID (SSE 기여 순)"]), "num")
    dd["비율"] = dd["발병주수"] / dd[C.Y2]
    odd = dd[(dd["비율"] < 120) | (dd["비율"] > 300)].sort_values("비율")
    t_odd = table(pd.DataFrame([[r.ID, f"{r[C.Y2]:.4f}", int(r['발병주수']), f"{r['비율']:.0f}"] for _, r in odd.iterrows()],
                               columns=["ID", "피해도", "발병주수", "발병주수 ÷ 피해도"]), "num")

    hur = pd.read_csv(f"{r2}/hurdle_summary.csv", header=[0, 1], index_col=0)
    hm, hs = hur.xs("mean", axis=1, level=1), hur.xs("std", axis=1, level=1)
    t_hur = table(pd.DataFrame([[lab] + [f"{hm.loc[m, c]:.3f} ± {hs.loc[m, c]:.3f}" for c in ["rho", "R2", "AUC_occ"]]
                                for m, lab in [("hurdle", "허들: P(발생) × 피해도|발생"), ("single", "단일 회귀 (142건)")]],
                               columns=["모형", "Spearman ρ", "R² (원척도)", "발생 판별 AUC"]), "num")

    c1, c3 = cfg["cls"], cfg["sev3"]
    body = BODY.format(
        n=len(d), n1=n1, n0=len(d) - n1,
        cls_auc=f'{c1["cv_nested"]:.3f}', sev_rho=f'{c3["cv_nested"]:.3f}',
        reg_r2=f'{diag["R2_mean"]:.3f}',
        t_del=t_del, t_cmp=t_cmp,
        cls_cfg=f'{c1["strategy"].upper()} · ExtraTrees · 변수 {c1["k"]}개 · 세트 {c1["set"]}',
        cls_bar=bar_chart(imp_rows(J["cls"])[:15], "shap"), leg_cls=LEG_RISK.format(w="발생 위험"),
        cls_forest=forest(eff["cls"]["table"]), cls_or=or_table(eff["cls"]["table"]), leg_or=LEG_OR,
        cls_core_auc=f'{eff["cls"]["cv_auc"][0]:.3f}',
        sev_counts=" / ".join(f"{a} {b}건" for a, b in zip(C.SEV_LABELS, eff["sev3"]["class_counts"])),
        sev_cfg=f'평균순위 · Random Forest · 변수 {c3["k"]}개', sev_acc=f'{J["sev3"]["perf_fixed"]["acc3"]["mean"]:.3f}',
        sev_bar=bar_chart(imp_rows(J["sev3"]), "shap"), leg_sev=LEG_RISK.format(w="피해"),
        sev_forest=forest(eff["sev3"]["table"]), sev_or=or_table(eff["sev3"]["table"]),
        t_tr=t_tr, t_tune=t_tune, t_coef=t_coef, reg_insample=f"{ols.rsquared:.3f}",
        t_rows=t_rows, t_sens=t_sens, t_odd=t_odd, t_hur=t_hur,
    )
    page = (f'<title>고추탄저병 재분석 v2</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'{FONTS}\n{STYLE}\n{body}')
    open(out, "w").write(page)
    print("wrote", out)


BODY = open(f"{HERE}/report_v2_body.html").read()

if __name__ == "__main__":
    main(*sys.argv[1:5])
