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


PR_LAB = {"제초매트": "제초매트", "보호_카_횟수": "보호살균제 횟수", "7~8월_사전경과_최대": "강우 전 방제 경과일",
          "7~8월_사후지연_최대": "강우 후 방제 지연"}
F_LAB = {"노출값": "노출값", "작년_탄저발생": "작년 발생", "연작": "연작", "인근고추밭_50m": "인근 고추밭",
         "log_성묘개수": "농가 규모(성묘 개수)", **PR_LAB}
# 변수값이 커질 때 발생 위험이 커지는(+1) / 작아지는(−1) 방향 (모형 구조상 정해짐)
F_DIR = {"노출값": 1, "작년_탄저발생": 1, "연작": 1, "인근고추밭_50m": 1, "log_성묘개수": -1, "제초매트": -1,
         "보호_카_횟수": -1, "7~8월_사전경과_최대": 1, "7~8월_사후지연_최대": 1}


def vul_fill(r4):
    r = json.load(open(f"{r4}/vulnerability/vulnerability.json"))
    pct = lambda v: f"{v * 100:.1f}"
    ci = lambda c: f"{c[0] * 100:.1f}–{c[1] * 100:.1f}"
    # 시나리오 행: 분석에 들어간 관행만 (run_vulnerability 의 PRACTICES)
    best_lab = {"7~8월_사전경과_최대": "강우 전 방제 경과일 ≤ 2일", "제초매트": "제초매트 사용",
                "보호_카_횟수": "보호살균제 10회 이상", "7~8월_사후지연_최대": "강우 후 방제 지연 ≤ 1일"}
    prs = list(r["params"]["w"])
    n_word = {2: "두", 3: "세", 4: "네"}.get(len(prs), str(len(prs)))
    keys = ([("현재 관행", "현재 관행")] + [(f"{p} 최선", best_lab[p]) for p in prs]
            + [("모두 최선 관행", f"{n_word} 가지 모두 최선")])

    def sc_table(sc, sci, red=None):
        rows = []
        for k, lab in keys:
            row = [lab]
            for e in ["0.1", "0.5", "0.9"]:
                cur = sc[e]["현재 관행"]
                v = sc[e][k]
                if k == "현재 관행":
                    row.append(f"{pct(v)}% ({ci(sci[e][k])})")
                else:
                    rr = (cur - v) / cur * 100
                    extra = f" [감소 {ci(red[e][k])}]" if red else ""
                    row.append(f"{pct(v)}% (−{(cur - v) * 100:.1f}%p, −{rr:.0f}%){extra}")
            rows.append(row)
        return table(pd.DataFrame(rows, columns=["재배방법 시나리오", "노출값 0.1", "노출값 0.5", "노출값 0.9"]), "num")

    t_sc = sc_table(r["scenarios"], r["scenarios_ci"], r["reduction_ci"])
    t_prev0 = sc_table(r["scenarios_prev"]["0"], r["scenarios_prev_ci"]["0"])
    t_prev1 = sc_table(r["scenarios_prev"]["1"], r["scenarios_prev_ci"]["1"])

    g = r["curve_grid"]
    idx = list(range(0, len(g), 2))
    series = {nm: [(f"{g[i]:.1f}", r["curve"][k][i]) for i in idx]
              for k, nm in [("모든 관행", f"{n_word} 가지 모두")] + [(p, PR_LAB[p]) for p in prs]}
    sh = r["shapley"]["평균|기여|"]
    sh_rows = sorted([{"var": F_LAB[k], "shap": v * 100, "rho": F_DIR[k]} for k, v in sh.items() if v > 1e-4],
                     key=lambda x: -x["shap"])
    t_sh = table(pd.DataFrame([[F_LAB[k], f"{sh[k] * 100:.1f}"] + [f"{r['shapley'][gname][k] * 100:+.1f}"
                                for gname in ["낮음(<0.3)", "중간(0.3–0.7)", "높음(≥0.7)"]]
                               for k in sorted(sh, key=lambda k: -sh[k])],
                              columns=["변수", "평균 |기여| (%p)", "노출 낮음 평균 (%p)", "노출 중간", "노출 높음"]), "num")
    inter = sorted(r["shapley_inter"].items(), key=lambda kv: -kv[1]["mean_abs"])[:6]
    t_int = table(pd.DataFrame([[" × ".join(F_LAB[x] for x in k.split("×")), f"{v['mean_abs'] * 100:.2f}",
                                 f"{v['mean'] * 100:+.2f}"] for k, v in inter],
                               columns=["변수 쌍", "평균 |결합 기여| (%p)", "평균 결합 기여 (%p)"]), "num")
    share = r["adapt_share"]
    w_rows = sorted([{"var": PR_LAB[k], "shap": v * 100, "rho": -1} for k, v in share.items()], key=lambda x: -x["shap"])
    t_w = table(pd.DataFrame([[PR_LAB[k], f"{r['params']['w'][k]:.2f}", f"{share[k] * 100:.0f}%",
                               f"{r['adapt_share_ci'][k][0] * 100:.0f}–{r['adapt_share_ci'][k][1] * 100:.0f}%",
                               {1: "클수록 좋음", -1: "작을수록 좋음"}[r["practice_ref"][k]["dir"]],
                               f"{r['practice_ref'][k]['low']:g} – {r['practice_ref'][k]['high']:g}"]
                              for k in sorted(share, key=lambda k: -share[k])],
                             columns=["재배방법", "효과 계수 w", "적응력 내 상대 기여", "부트스트랩 95% 구간",
                                      "방향", "점수화 범위 (0 ↔ 1)"]), "num")
    cv = r["cv"]
    t_cv = table(pd.DataFrame([[k, f"{v['AUC'][0]:.3f} ± {v['AUC'][1]:.3f}", f"{v['logloss']:.3f}"] for k, v in cv.items()],
                              columns=["모형", "CV AUC", "CV 로그손실"]), "num")
    t_auc = table(pd.DataFrame([["노출값만", f"{r['AUC_E']:.3f}"], ["1 − 적응력만", f"{r['AUC_A']:.3f}"],
                                ["노출값 × (1 − 적응력)", f"{r['AUC_V_simple']:.3f}"],
                                ["노출값 × 민감도 × (1 − 적응력)", f"{r['AUC_V']:.3f}"],
                                ["유효 노출 × 민감도 × (1 − 적응력) (모형 구조 그대로)", f"{r['AUC_V_eff']:.3f}"]],
                               columns=["지수", "발생 판별 AUC (학습 자료)"]), "num")
    p = r["params"]
    return {"v_t_sc": t_sc, "v_t_prev0": t_prev0, "v_t_prev1": t_prev1,
            "v_curve": line_chart(series, "발생확률 감소", xlab="노출값"),
            "v_sh_bar": bar_chart(sh_rows, "shap", fmt="{:.1f}"), "v_t_sh": t_sh, "v_t_int": t_int,
            "v_w_bar": bar_chart(w_rows, "shap", fmt="{:.0f}"), "v_t_w": t_w, "v_t_cv": t_cv, "v_t_auc": t_auc,
            "v_alpha": f"{p['alpha']:.2f}", "v_thx": f"{p['theta_x']:.2f}", "v_A_mean": f"{r['A_mean']:.2f}",
            "v_base": f"{r['shapley_base'] * 100:.1f}", **formula_fill(r), **adapt_table(r4)}


def formula_fill(r):
    """적응력·민감도 계산식을 추정값으로 만든다."""
    p = r["params"]
    sub = {"7~8월_사전경과_최대": "강우전경과", "제초매트": "제초매트", "보호_카_횟수": "보호살균제",
           "7~8월_사후지연_최대": "강우후지연"}
    a_terms = "".join(f"(1 − {w:.2f}·a<sub>{sub[k]}</sub>)" for k, w in p["w"].items())
    th = p["theta"]
    s_terms = (f"{th['작년_탄저발생']:.2f}·작년발생 + " if th["작년_탄저발생"] > 0.005 else "") + \
        f"{th['연작']:.2f}·연작 + {th['인근고추밭_50m']:.2f}·인근고추밭 − {abs(p['beta_size']):.2f}·규모<sub>z</sub>"
    return {"v_A_formula": f"A = 1 − {a_terms}", "v_S_formula": s_terms,
            "v_n_prac": str(len(p["w"])), "v_npairs": str(len(p["w"]) * (len(p["w"]) - 1) // 2)}


def adapt_table(r4):
    """적응력 요소 후보 조합 비교표 (run_adapt_compare)."""
    f = f"{r4}/vulnerability/adapt_compare.json"
    if not os.path.exists(f):
        return {"v_t_adapt": ""}
    c = json.load(open(f))
    rows = []
    for name in sorted(k for k in c if k[:1] == "S"):
        v = c[name]
        ws = ", ".join(f"{PR_LAB.get(k, '방제강도 지수' if k == '방제강도_지수' else ('7월 살포횟수' if k == '7월_살포횟수' else ('살포일수' if k == '살포일수' else k)))} {w:.2f}"
                       f"{' ✓' if v['w_ci'][k][0] > 0 else ''}" for k, w in v["w"].items())
        rows.append([name.split(" ", 1)[0], name.split(" ", 1)[1], ws,
                     f"{v['cv']['B 곱셈형']['AUC'][0]:.3f}", f"{v['cv']['A 로지스틱+E×관행']['AUC'][0]:.3f}",
                     f"{v['cv']['B 곱셈형']['logloss']:.3f}", f"{v['AIC']:.1f}"])
    return {"v_t_adapt": table(pd.DataFrame(rows, columns=["조합", "적응력 후보", "가중치 w (✓ = 95% 구간이 0 제외)",
                                                          "CV AUC (곱셈형)", "CV AUC (로지스틱)", "CV 로그손실",
                                                          "AIC"]), "num"),
            "v_int_load": ", ".join(f"{k} {v:.2f}" for k, v in c.get("intensity_loadings", {}).items()),
            "v_int_evr": f"{c.get('intensity_evr', 0) * 100:.0f}"}


def main(x4, r4, x3, r3, out, body_file="report_v4_body.html", title="고추탄저병 최종 분석", tr=None):
    """tr: 생성된 표·그림 문자열에 적용할 번역 함수 (영문 보고서용)."""
    # 피해도 회귀 섹션: v4 회귀 결과가 있으면 그것을, 없으면 v3 결과를 사용
    # 초발 시기 변수(피해 진행과 같은 성격)를 뺀 재분석(reg_noonset)을 우선 사용
    reg_dir = next((f"{r4}/{sub}" for sub in ["reg_noonset", "reg"] if os.path.exists(f"{r4}/{sub}/reg3_et7.json")), r3)
    reg_x = x4 if reg_dir != r3 else x3
    # 회귀만 더 새 자료로 다시 돌린 경우: REG_XLSX=<엑셀> REG_DIR=<결과폴더>
    reg_dir, reg_x = os.environ.get("REG_DIR", reg_dir), os.environ.get("REG_XLSX", reg_x)
    fill = build_fill(reg_x, reg_dir)
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
        # 오즈비: 사전 지정 모형(run_prespec)이 있으면 그것을, 없으면 이전 핵심변수 모형을 사용
        pf = f"{r4}/{name}/prespec.json"
        if os.path.exists(pf):
            e = json.load(open(pf))
            e = {**e["cls"], "_pre": e}
        else:
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
                                        "사전 지정 로지스틱 AUC (Firth)"]), "num")

    def stab_table(e, n_other=6):
        pre = e.get("_pre")
        if not pre:
            return ""
        freq, roles = pre["stability"]["freq"], pre["roles"]
        level = lambda f: "강함" if f >= 0.7 else ("중간" if f >= 0.4 else "약함")
        rows = [[v, roles[v], f"{freq.get(v, 0):.2f}", level(freq.get(v, 0)), "포함"]
                for v in sorted(pre["spec"], key=lambda v: -freq.get(v, 0))]
        others = [v for v in freq if v not in pre["spec"]][:n_other]
        rows += [[v, "-", f"{freq[v]:.2f}", level(freq[v]), "-"] for v in others]
        return table(pd.DataFrame(rows, columns=["변수", "역할", "안정성 선택 빈도", "근거 강도", "사전 지정 모형"]), "num")

    def cls_fill(name, pre):
        c, j, e = V[name]
        pp = e.get("_pre", {})
        st = pp.get("stability", {})
        return {f"{pre}_stab": stab_table(e), f"{pre}_epv": f'{pp.get("epv", 0):.1f}',
                f"{pre}_nspec": str(len(pp.get("spec", []))), f"{pre}_ev": f'{st.get("EV_bound", 0):.1f}',
                f"{pre}_ncand": str(st.get("n_candidates", 0)),f"{pre}_cfg": f'{STRAT[c["strategy"]]}로 고른 변수 {c["k"]}개 · {MODEL[c["model"]]}',
                f"{pre}_auc": f'{c["cv_nested"]:.3f}', f"{pre}_auc_fixed": f'{j["perf_fixed"]["AUC"]["mean"]:.3f}',
                f"{pre}_bar": bar_chart(imp_rows(j)[:15], "shap"), f"{pre}_forest": forest(e["table"]),
                f"{pre}_or": or_table(e["table"]), f"{pre}_core_auc": f'{e["cv_auc"][0]:.3f}',
                f"{pre}_pr2": f'{e["pseudoR2"]:.3f}'}

    fill.update(cls_fill("all", "ca"))
    fill.update(cls_fill("noweather", "cn"))
    fill.update(vul_fill(r4))
    fill.update(n=len(d4), n1=npos, n0=len(d4) - npos, thr=f"{t:g}", thr_pct=f"{t * 100:g}",
                thr_curve=line_chart(series, "AUC", xlab="임계값"), t_thr=t_thr, t_var=t_var,
                nt_auc=f'{cfg["noweather_nt"]["cv_nested"]:.3f}', leg_cls=LEG_RISK.format(w="발생 위험"),
                leg_or=LEG_OR)
    body = open(f"{HERE}/{body_file}").read()
    for k, v in fill.items():
        body = body.replace("{{" + k + "}}", tr(str(v)) if tr else str(v))
    page = (f'<title>{title}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f"{FONTS}\n{STYLE}\n{EXTRA_CSS}\n{body}")
    open(out, "w").write(page)
    print("wrote", out, "unfilled:", re.findall(r"\{\{\w+\}\}", page))


if __name__ == "__main__":
    main(*sys.argv[1:6])
