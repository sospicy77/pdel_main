"""results/ 의 산출물로 HTML 보고서(report.html)를 만든다.

사용법: python make_report.py <엑셀경로> <results 폴더> <출력 html>
"""
import html
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

E = html.escape


# ---------------- SVG 차트 ----------------
def dir_class(rho):
    return "up" if rho > 0.2 else ("down" if rho < -0.2 else "flat")


DIR_LABEL = {"up": "값↑ → 증가", "down": "값↑ → 감소", "flat": "비단조"}


def bar_chart(rows, value_key, fmt="{:.3f}", width=640, label_w=190):
    """가로 막대. rows: dict(var, value, rho). 색 = 효과 방향."""
    bh, gap = 20, 6
    h = len(rows) * (bh + gap) + 28
    raw = max(r[value_key] for r in rows) or 1
    step = [m * 10 ** e for e in range(-4, 2) for m in (1, 2, 5) if raw / (m * 10 ** e) <= 5][0]
    vmax = math.ceil(raw / step) * step
    plot_w = width - label_w - 70
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    for t in np.arange(0, vmax + step / 2, step):
        x = label_w + t / vmax * plot_w
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{h-24}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{h-8}" class="tick" text-anchor="middle">{t:g}</text>')
    for i, r in enumerate(rows):
        y = i * (bh + gap)
        w = max(2, r[value_key] / vmax * plot_w)
        c = dir_class(r["rho"])
        tip = f'{r["var"]}: {fmt.format(r[value_key])} · {DIR_LABEL[c]} (ρ={r["rho"]:.2f})'
        out.append(f'<g class="mark"><title>{E(tip)}</title>'
                   f'<rect x="0" y="{y}" width="{width}" height="{bh+gap}" class="hit"/>'
                   f'<text x="{label_w-8}" y="{y+bh/2+4}" class="lab" text-anchor="end">{E(r["var"])}</text>'
                   f'<path d="M{label_w},{y} h{w-4:.1f} q4,0 4,4 v{bh-8} q0,4 -4,4 h-{w-4:.1f} z" class="bar {c}"/>'
                   f'<text x="{label_w+w+6:.1f}" y="{y+bh/2+4}" class="val">{fmt.format(r[value_key])}</text></g>')
    out.append("</svg>")
    return "".join(out)


def forest(rows, width=640, label_w=190):
    """오즈비 forest plot (로그축)."""
    bh = 26
    h = len(rows) * bh + 34
    lo = min(min(r["lo"] for r in rows), 0.5)
    hi = max(max(r["hi"] for r in rows), 2)
    lo, hi = math.log(lo), math.log(hi)
    plot_w = width - label_w - 110
    X = lambda v: label_w + (math.log(v) - lo) / (hi - lo) * plot_w
    ticks = [t for t in [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20] if lo <= math.log(t) <= hi]
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    for t in ticks:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="0" y2="{h-26}" class="{"ref" if t == 1 else "grid"}"/>'
                   f'<text x="{X(t):.1f}" y="{h-10}" class="tick" text-anchor="middle">{t:g}</text>')
    for i, r in enumerate(rows):
        y = i * bh + bh / 2
        c = "up" if r["lo"] > 1 else ("down" if r["hi"] < 1 else "flat")
        sig = "*" if r["p"] < 0.05 else ""
        tip = f'{r["var"]} ({r["unit"]}): OR {r["OR"]:.2f} [{r["lo"]:.2f}–{r["hi"]:.2f}], p={r["p"]:.3f}'
        out.append(f'<g class="mark"><title>{E(tip)}</title>'
                   f'<rect x="0" y="{y-bh/2}" width="{width}" height="{bh}" class="hit"/>'
                   f'<text x="{label_w-8}" y="{y+4}" class="lab" text-anchor="end">{E(r["var"])}</text>'
                   f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y}" y2="{y}" class="ci {c}"/>'
                   f'<circle cx="{X(r["OR"]):.1f}" cy="{y}" r="5" class="pt {c}"/>'
                   f'<text x="{label_w+plot_w+10}" y="{y+4}" class="val">{r["OR"]:.2f}{sig}</text></g>')
    out.append("</svg>")
    return "".join(out)


def line_chart(series, ylab, width=640, height=260, xlab="변수 수 k"):
    """series: {name: [(k, v), ...]}. 최대 3개 계열."""
    # 오른쪽 여백은 끝 라벨(계열 이름) 길이에 맞춤
    pad_l, pad_b, pad_t = 44, 30, 10
    pad_r = max(120, 7 * max(len(n) for n in series) + 24)
    ks = sorted({k for s in series.values() for k, _ in s})
    vs = [v for s in series.values() for _, v in s]
    step = 0.02 if max(vs) - min(vs) < 0.15 else 0.05
    vmin, vmax = math.floor(min(vs) / step) * step, math.ceil(max(vs) / step) * step
    xi = {k: i for i, k in enumerate(ks)}
    X = lambda k: pad_l + xi[k] / (len(ks) - 1) * (width - pad_l - pad_r)
    Y = lambda v: pad_t + (vmax - v) / (vmax - vmin) * (height - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for t in np.arange(vmin, vmax + 1e-9, step):
        out.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                   f'<text x="{pad_l-6}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{t:.2f}</text>')
    for k in ks:
        out.append(f'<text x="{X(k):.1f}" y="{height-10}" class="tick" text-anchor="middle">{k}</text>')
    out.append(f'<text x="{width-pad_r}" y="{height-10}" class="tick" text-anchor="start" dx="14">{xlab}</text>')
    for i, (name, s) in enumerate(series.items()):
        pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in s)
        out.append(f'<polyline points="{pts}" class="line s{i+1}"/>')
        for k, v in s:
            out.append(f'<g class="mark"><title>{E(name)} · k={k}: {ylab} {v:.3f}</title>'
                       f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="10" class="hit"/>'
                       f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="4" class="dot s{i+1}"/></g>')
    ends = sorted(((Y(s[-1][1]), name) for name, s in series.items()))
    last = -99
    for y, name in ends:
        y = max(y, last + 14)
        last = y
        out.append(f'<text x="{X(ks[-1])+12:.1f}" y="{y+4:.1f}" class="lab">{E(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def dependence(x, s, name, width=300, height=170):
    """SHAP 의존 산점도 (작은 다중그림 1칸)."""
    pad_l, pad_b, pad_t, pad_r = 40, 26, 22, 10
    x, s = np.asarray(x, float), np.asarray(s, float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        xmax = xmin + 1
    a = max(abs(s.min()), abs(s.max())) * 1.1
    X = lambda v: pad_l + (v - xmin) / (xmax - xmin) * (width - pad_l - pad_r)
    Y = lambda v: pad_t + (a - v) / (2 * a) * (height - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart small" role="img">',
           f'<text x="{pad_l}" y="14" class="lab strong">{E(name)}</text>',
           f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="ref"/>',
           f'<text x="{pad_l-4}" y="{Y(a*0.8)+4:.1f}" class="tick" text-anchor="end">+{a*0.8:.2f}</text>',
           f'<text x="{pad_l-4}" y="{Y(-a*0.8)+4:.1f}" class="tick" text-anchor="end">−{a*0.8:.2f}</text>']
    for t in np.linspace(xmin, xmax, 3):
        out.append(f'<text x="{X(t):.1f}" y="{height-8}" class="tick" text-anchor="middle">{t:.3g}</text>')
    for xv, sv in zip(x, s):
        out.append(f'<circle cx="{X(xv):.1f}" cy="{Y(sv):.1f}" r="3.2" class="pt2 {"up" if sv > 0 else "down"}">'
                   f'<title>{E(name)}={xv:.3g} → SHAP {sv:+.3f}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def table(df, cls=""):
    head = "".join(f"<th>{E(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{E(str(v))}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------------- 본문 ----------------
def main(xlsx, rd, out):
    d = C.load(xlsx)
    J = {t: json.load(open(f"{rd}/{t}_final.json")) for t in ["cls", "sev3", "reg"]}
    eff = json.load(open(f"{rd}/effects.json"))
    hurdle = pd.read_csv(f"{rd}/hurdle_summary.csv", header=[0, 1], index_col=0) \
        if os.path.exists(f"{rd}/hurdle_summary.csv") else None

    # 1단계 변수세트 비교
    s1 = []
    for t, m, lab in [("cls", "AUC", "발생 분류 (AUC)"), ("sev3", "rho", "피해도 3단계 (ρ)"),
                      ("reg", "R2", "피해도 회귀 (R²)")]:
        s = pd.read_csv(f"{rd}/{t}_cv_summary.csv")
        best = s.loc[s.groupby("set")[f"{m}_mean"].idxmax()].set_index("set")
        s1.append([lab] + [f'{best.loc[k, f"{m}_mean"]:.3f} ({best.loc[k, "model"]}, k={best.loc[k, "k"]})'
                           for k in ["A", "A_bin", "B"]])
    t_sets = pd.DataFrame(s1, columns=["과제", "A: v7 + 노출값", "A_bin: 노출값 4등급", "B: A + 기상8·IR2"])

    # 2단계: 발생 분류 선택방법 곡선 (각 k 에서 가장 좋은 모델)
    s2 = pd.read_csv(f"{rd}/cls_B_sel2_summary.csv")
    s2 = s2[s2.k != "var"].assign(k=lambda x: x.k.astype(int))
    names = {"meanrank": "평균순위", "rfshap": "RF-SHAP", "rfe": "RF-RFE"}
    series = {names[st]: [(k, g[g.k == k].AUC_mean.max()) for k in sorted(g.k.unique())]
              for st, g in s2.groupby("strategy")}
    bor = pd.read_csv(f"{rd}/cls_B_sel2_summary.csv").query("strategy=='boruta'").AUC_mean.max()

    best2 = []
    for t, sn, m in [("cls", "B", "AUC"), ("sev3", "A", "rho"), ("reg", "A", "R2")]:
        s = pd.read_csv(f"{rd}/{t}_{sn}_sel2_summary.csv")
        for st in ["meanrank", "rfshap", "rfe", "boruta"]:
            g = s[s.strategy == st].iloc[0]
            best2.append([{"cls": "발생", "sev3": "피해도 3단계", "reg": "피해도 회귀"}[t],
                          {**names, "boruta": "Boruta"}[st], g.model, g.k,
                          f'{g[f"{m}_mean"]:.3f} ± {g[f"{m}_std"]:.3f}'])
    t_sel = pd.DataFrame(best2, columns=["과제", "선택 방법", "최고 모델", "k", "성능 (평균 ± SD)"])

    def imp_rows(j):
        return [{"var": r["var"], "shap": r["mean_abs_shap"], "rho": r["direction_rho"],
                 "freq": r["boot_freq"], "perm": r["perm_mean"]} for r in j["importance"]]

    def imp_table(j, permlab):
        return table(pd.DataFrame([[i + 1, r["var"], f'{r["mean_abs_shap"]:.3f}',
                                    DIR_LABEL[dir_class(r["direction_rho"])],
                                    f'{r["perm_mean"]:+.3f}', f'{r["boot_freq"]*100:.0f}%']
                                   for i, r in enumerate(j["importance"])],
                                  columns=["순위", "변수", "mean|SHAP|", "방향", permlab, "부트스트랩 선택률"]),
                     "num")

    # 발생 SHAP 의존도 (상위 6개)
    sv = pd.read_csv(f"{rd}/cls_final_shap_values.csv")
    Xc = pd.read_csv(f"{rd}/cls_final_X.csv")
    top6 = [r["var"] for r in J["cls"]["importance"][:6]]
    deps = "".join(f'<figure class="dep">{dependence(Xc[v], sv[v], v)}</figure>' for v in top6)

    def or_table(rows):
        return table(pd.DataFrame([[r["var"], r["unit"], f'{r["OR"]:.2f}', f'{r["lo"]:.2f}–{r["hi"]:.2f}',
                                    f'{r["p"]:.3f}'] for r in rows],
                                  columns=["변수", "단위", "OR", "95% CI", "p"]), "num")

    perf = {t: J[t]["perf_fixed"] for t in J}
    hur_html = ""
    if hurdle is not None:
        hm = hurdle.xs("mean", axis=1, level=1)
        hs = hurdle.xs("std", axis=1, level=1)
        rows = []
        for m, lab in [("hurdle", "허들: P(발생) × 피해도|발생"), ("single", "단일 회귀: logit(Y2), 150건")]:
            rows.append([lab] + [f"{hm.loc[m, c]:.3f} ± {hs.loc[m, c]:.3f}" for c in ["rho", "R2", "R2_log", "AUC_occ"]])
        hur_html = table(pd.DataFrame(rows, columns=["모형", "Spearman ρ", "R² (원척도)",
                                                     "R² (log10)", "발생 판별 AUC"]), "num")
        hwin = "허들 모형" if hm.loc["hurdle", "rho"] > hm.loc["single", "rho"] else "단일 회귀"
    else:
        hwin = None

    # 피해도 회귀 개선 (run_reg_tune / run_reg_diag)
    ta = pd.read_csv(f"{rd}/regtune_A_A.csv")
    tlab = {"logit": "logit (기존)", "log10": "log10", "sqrt": "제곱근", "asin": "arcsine √", "raw": "원값"}
    t_tr = pd.DataFrame([[tlab[t], g.iloc[0].model, g.iloc[0].k, f'{g.iloc[0].R2_t_mean:.3f}',
                          f'{g.iloc[0].R2_raw_mean:.3f}', f'{g.iloc[0].rho_mean:.3f}']
                         for t, g in ta.groupby("transform", sort=False)],
                        columns=["변환", "최고 모델", "k", "R² (변환 척도)", "R² (원척도)", "Spearman ρ"])
    tb = pd.read_csv(f"{rd}/regtune_B_log10_A.csv", index_col=0)
    tc = pd.read_csv(f"{rd}/regtune_C_log10_A.csv", index_col=0)
    grid = [l.rsplit(") ", 1) for l in open(f"{rd}/regtune_etgrid.txt").read().splitlines()]
    grid = sorted(grid, key=lambda g: -float(g[1].split()[0]))
    tune_rows = [["nested 튜닝 (k·하이퍼파라미터 모두 안쪽 CV)", m, f'{tb.loc[m, "R2_t_mean"]:.3f} ± {tb.loc[m, "R2_t_std"]:.3f}']
                 for m in tb.sort_values("R2_t_mean", ascending=False).index]
    tune_rows += [["최적 부분집합 (PRESS 최소, 후보 14개 중 ≤6개)", m, f'{tc.loc[m, "R2_t_mean"]:.3f} ± {tc.loc[m, "R2_t_std"]:.3f}']
                  for m in tc.index]
    tune_rows += [["ExtraTrees 격자 탐색 (k, 잎 최소표본, max_features)", f"{g[0]})", g[1]] for g in grid[:3]]
    t_tune = pd.DataFrame(tune_rows, columns=["방식", "모델 / 설정", "R² (log10, 평균 ± SD)"])
    imp_f = pd.read_csv(f"{rd}/regfinal_log10_importance.csv", index_col=0)
    rt_bar = bar_chart([{"var": v, "shap": r.mean_abs_shap, "rho": r.rho} for v, r in imp_f.iterrows()], "shap")
    rows_d = pd.read_csv(f"{rd}/regdiag_log10_A_rows.csv").head(15)
    t_rows = table(pd.DataFrame([[i + 1, r.ID, f"{r.피해도:.4f}", int(r.발병주수), f"{r.예측_피해도:.3f}",
                                  f"{r.잔차:+.2f}", f"{r.SSE기여율*100:.1f}%", r.방향.split("(")[0],
                                  r.같은농가_피해도범위] for i, r in enumerate(rows_d.itertuples())],
                                columns=["순위", "ID", "실제 피해도", "발병주수", "예측 피해도", "잔차 (log10)",
                                         "SSE 기여", "유형", "같은 농가 피해도 범위"]), "num")
    dg = json.load(open(f"{rd}/regdiag_log10_A.json"))
    t_sens = table(pd.DataFrame([[s["제외_농가수"], f'{s["R2"]:.3f}', s["제외_ID"] or "-"] for s in dg["sensitivity"]],
                                columns=["제외 농가 수", "R² (log10)", "제외한 ID (SSE 기여 순)"]), "num")

    n1 = int(d[C.Y1].sum())
    fill = dict(
        rt_transform=table(t_tr, "num"), rt_tune=table(t_tune, "num"), rt_bar=rt_bar,
        rt_rows=t_rows, rt_sens=t_sens, rt_r2=f'{dg["R2_mean"]:.3f}',
        n=len(d), n1=n1, n0=len(d) - n1,
        auc_nested="0.772", auc_fixed=f'{perf["cls"]["AUC"]["mean"]:.3f}',
        acc_nested="0.642", rho_sev_nested="0.617", acc_fixed=f'{perf["sev3"]["acc3"]["mean"]:.3f}',
        r2_nested="0.148", r2_fixed=f'{perf["reg"]["R2"]["mean"]:.3f}',
        t_sets=table(t_sets),
        curve=line_chart(series, "AUC"), boruta=f"{bor:.3f}",
        t_sel=table(t_sel, "num"),
        cls_k=len(J["cls"]["selected"]),
        cls_bar=bar_chart(imp_rows(J["cls"]), "shap"),
        cls_tbl=imp_table(J["cls"], "Permutation ΔAUC"),
        deps=deps,
        cls_forest=forest(eff["cls"]["table"]), cls_or=or_table(eff["cls"]["table"]),
        cls_r2=f'{eff["cls"]["pseudoR2"]:.3f}', cls_core_auc=f'{eff["cls"]["cv_auc"][0]:.3f}',
        sev_counts=" / ".join(f"{a} {b}건" for a, b in zip(C.SEV_LABELS, eff["sev3"]["class_counts"])),
        sev_bar=bar_chart(imp_rows(J["sev3"]), "shap"),
        sev_tbl=imp_table(J["sev3"], "Permutation Δρ"),
        sev_forest=forest(eff["sev3"]["table"]), sev_or=or_table(eff["sev3"]["table"]),
        reg_bar=bar_chart(imp_rows(J["reg"]), "shap"),
        reg_tbl=imp_table(J["reg"], "Permutation ΔR²"),
        hurdle=hur_html or "<p>허들 결과 파일이 없습니다.</p>",
        hwin=hwin or "-",
    )
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", str(v))
    open(out, "w").write(page)
    print("wrote", out)


TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")).read()

if __name__ == "__main__":
    main(*sys.argv[1:4])
