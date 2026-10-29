"""보고서 맨 앞의 분석 파이프라인 다이어그램 (인라인 SVG, 국문·영문).

세 갈래(발생 · 취약성 · 피해도)를 위에서 아래로 그리고, 단계 종류를 색으로 구분한다.
  ML    : 변수 선택·모델 비교·성능 검증·SHAP 해석·오차 진단 (accent 채움)
  통계  : 효과 크기·감염압 모형·시나리오·기여도 분해 (중립)
  설계  : 분석 규칙 (점선)

사용법: python pipeline_diagram.py  → pipeline_kr.svg.html, pipeline_en.svg.html (본문에 붙여 넣을 조각)
"""
import os
from html import escape as E

W, COLW, X = 900, 250, [20, 325, 630]
ROW0, PITCH, BH = 164, 76, 56

T = {
    "kr": {
        "data": ("설문 자료 (v4) · 141개 농가", "재배·방제 관행 · 민감도(작년 발생·연작·인근 고추밭) · 기상 10개 · 노출값 · 피해도(Y2)"),
        "heads": [("발생: 피해도 > 3%", "141농가 · 02–05장"), ("취약성: E × S × (1 − A)", "141농가 · 06장"),
                  ("피해 정도: log10 Y2", "발병 60농가 · 07–09장")],
        "tag": {"ml": "ML", "st": "통계", "rule": "설계"},
        "A": [("ml", "발생 기준 선정", "임계값 0–3% × 모델 4종", "CV AUC 최고 → 3%"),
              ("ml", "변수 선택 + 모델 비교", "선택법 4 × 모델 6 × k", "fold 안에서 선택 · nested CV"),
              ("ml", "해석: SHAP · 안정성 선택", "중요 변수와 위험 방향", "후보 변수의 근거 강도"),
              ("st", "효과 크기: 사전 지정 모형", "Firth 로지스틱", "오즈비 · 95% CI · p"),
              ("ml", "기상 변수 조합 선별", "45개 조합 × RF", "nested CV로 낙관 확인")],
        "B": {1: ("rule", "취약성 틀 정의", "E 노출값 · S 민감도", "A 재배 관행"),
              3: ("ml", "적응력 후보 조합", "SHAP 상위 관행 포함", "9개 조합"),
              4: ("st", "곱셈형 감염압 모형", "CV AUC · AIC 비교", "→ 적응력 관행 2개"),
              5: ("st", "시나리오 (g-computation)", "노출값 0.1 / 0.5 / 0.9", "관행별 발생확률 감소"),
              6: ("st", "기여도 분해 (Shapley)", "변수별 기여 %p", "적응력 안 관행 비중")},
        "C": [("rule", "초발 시기 변수 제외", "초발_DOY · 정식_초발_경과일", "반응변수와 같은 성격"),
              ("ml", "회귀 모델 15종 비교", "k = 3 … 전체 · log10 / 원척도", "반복 5-fold CV"),
              ("ml", "nested CV → 최종 모델", "선택 편향을 뺀 성능", "ExtraTrees (변수 15개)"),
              ("ml", "해석: SHAP · permutation", "변수별 효과 방향·크기", "fold 선택률"),
              ("st", "해석용 OLS", "핵심 변수 3개", "→ 피해도 배수 · CI"),
              ("ml", "오차 큰 농가 진단", "out-of-fold 잔차", "→ 재조사 후보")],
        "outs": [("발생 위험 요인", "6월 기온·7~8월 강수량 ↑", "강우 직전 방제·제초매트 ↓"),
                 ("관행별 효과와 취약성 지수", "강우 직전 방제·제초매트", "발생확률 감소량 (%p)"),
                 ("피해도를 좌우한 변수", "이환과 즉시 제거 ↓", "작년 발생 ↑")],
        "concl": "결론 (10장): 발생 기준 · 위험 요인 · 기상 정보 · 취약성 · 피해 정도",
        "x_shap": "SHAP", "x_shap2": "순위", "x_eas": "변수 역할", "x_eas2": "사전 지정",
        "legend": [("ml", "ML: 변수 선택 · 모델 비교 · 검증 · 해석 · 진단"), ("st", "통계 모형: 효과 크기 · 시나리오"),
                   ("rule", "분석 설계 규칙")],
        "aria": "분석 파이프라인. 설문 자료에서 발생, 취약성, 피해도 세 갈래로 나뉘고, ML은 발생 기준 선정, 변수 선택과 모델 비교, "
                "SHAP 해석, 기상 조합 선별, 회귀 모델 비교와 nested CV, 오차 진단에 쓰였으며, 효과 크기와 취약성 시나리오는 통계 모형으로 추정했다.",
    },
    "en": {
        "data": ("Survey data (v4) · 141 farms", "Cropping & spray practices · sensitivity (disease last year, continuous cropping, nearby field) · 10 weather · exposure · severity (Y2)"),
        "heads": [("Occurrence: severity > 3%", "141 farms · Sections 02–05"), ("Vulnerability: E × S × (1 − A)", "141 farms · Section 06"),
                  ("Severity: log10 Y2", "60 diseased farms · Sections 07–09")],
        "tag": {"ml": "ML", "st": "Stats", "rule": "Design"},
        "A": [("ml", "Occurrence threshold", "Thresholds 0–3% × 4 models", "Best CV AUC → 3%"),
              ("ml", "Variable selection + models", "4 selectors × 6 models × k", "Selection in fold · nested CV"),
              ("ml", "Interpretation: SHAP", "Key variables & risk direction", "Stability selection"),
              ("st", "Effect sizes: pre-specified", "Firth logistic regression", "Odds ratios · 95% CI · p"),
              ("ml", "Weather pair screen", "45 pairs × RF", "Nested CV checks optimism")],
        "B": {1: ("rule", "Vulnerability framework", "E exposure · S sensitivity", "A cropping practices"),
              3: ("ml", "Adaptive candidate sets", "Incl. top SHAP practices", "9 candidate sets"),
              4: ("st", "Multiplicative hazard model", "CV AUC · AIC comparison", "→ 2 adaptive practices"),
              5: ("st", "Scenarios (g-computation)", "Exposure 0.1 / 0.5 / 0.9", "Risk reduction by practice"),
              6: ("st", "Contributions (Shapley)", "Per-variable pp", "Share within adaptive capacity")},
        "C": [("rule", "Drop onset-timing variables", "First-symptom DOY, interval", "Same nature as response"),
              ("ml", "Compare 15 regressors", "k = 3 … all · log10 / raw", "Repeated 5-fold CV"),
              ("ml", "Nested CV → final model", "Performance without bias", "ExtraTrees (15 variables)"),
              ("ml", "Interpretation: SHAP", "Permutation importance", "Fold selection rate"),
              ("st", "Interpretive OLS", "3 core variables", "→ severity multipliers · CI"),
              ("ml", "Large-error farms", "Out-of-fold residuals", "→ re-survey candidates")],
        "outs": [("Occurrence risk factors", "June temp, Jul–Aug rain ↑", "Spray before rain, weed mat ↓"),
                 ("Practice effects & index", "Spray before rain, weed mat", "Risk reduction (pp)"),
                 ("Drivers of severity", "Immediate fruit removal ↓", "Disease last year ↑")],
        "concl": "Conclusions (Section 10): threshold · risk factors · weather · vulnerability · severity",
        "x_shap": "SHAP", "x_shap2": "rank", "x_eas": "E·S·A", "x_eas2": "roles",
        "legend": [("ml", "ML: selection · comparison · validation · SHAP"),
                   ("st", "Statistical model: effect sizes · scenarios"), ("rule", "Design rule")],
        "aria": "Analysis pipeline. Survey data splits into occurrence, vulnerability and severity branches. ML was used for the occurrence "
                "threshold, variable selection and model comparison, SHAP interpretation, weather-pair screening, regression model comparison "
                "with nested CV, and error diagnostics; effect sizes and vulnerability scenarios come from statistical models.",
    },
}

STY = {"ml": 'style="fill:var(--accent-soft);stroke:var(--accent)" stroke-width="1.5"',
       "st": 'style="fill:var(--surface);stroke:var(--muted)" stroke-width="1"',
       "rule": 'style="fill:var(--surface);stroke:var(--muted)" stroke-width="1" stroke-dasharray="4 3"'}
TAGF = {"ml": "fill:var(--accent)", "st": "fill:var(--muted)", "rule": "fill:var(--muted)"}


def ry(r):
    return ROW0 + (r - 1) * PITCH


def box(x, y, kind, title, s1, s2, tag, w=COLW):
    tw = 12 + 7 * len(tag)
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="6" {STY[kind]}/>'
            f'<rect x="{x + w - tw - 6}" y="{y + 6}" width="{tw}" height="15" rx="7.5" style="{TAGF[kind]}"/>'
            f'<text x="{x + w - tw / 2 - 6}" y="{y + 17}" text-anchor="middle" font-size="10" font-weight="600" '
            f'style="fill:var(--surface)">{E(tag)}</text>'
            f'<text x="{x + 10}" y="{y + 19}" font-size="12.5" font-weight="600">{E(title)}</text>'
            f'<text x="{x + 10}" y="{y + 35}" font-size="11" style="fill:var(--muted)">{E(s1)}</text>'
            f'<text x="{x + 10}" y="{y + 49}" font-size="11" style="fill:var(--muted)">{E(s2)}</text>')


def vline(x, y1, y2, ml=False):
    m = "url(#pa-a)" if ml else "url(#pa)"
    st = 'style="stroke:var(--accent)"' if ml else 'stroke="currentColor"'
    return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 2}" {st} stroke-width="1.2" marker-end="{m}"/>'


def svg(lang):
    t = T[lang]
    o = []
    concl_y = ry(7) + BH + 28
    leg_y = concl_y + 44 + 30
    H = leg_y + 20
    o.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{E(t["aria"])}" class="pipeline" '
             f'style="color:var(--fg);font-family:var(--f-body)" fill="currentColor">')
    o.append('<defs><marker id="pa" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker>'
             '<marker id="pa-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L8,4 L0,8 z" style="fill:var(--accent)"/></marker></defs>')
    # 자료
    o.append(f'<rect x="20" y="16" width="{W - 40}" height="56" rx="6" style="fill:var(--surface);stroke:var(--fg)" stroke-width="1.2"/>')
    o.append(f'<text x="{W / 2}" y="39" text-anchor="middle" font-size="13.5" font-weight="700">{E(t["data"][0])}</text>')
    o.append(f'<text x="{W / 2}" y="58" text-anchor="middle" font-size="11" style="fill:var(--muted)">{E(t["data"][1])}</text>')
    # 갈래 머리
    for i, (h1, h2) in enumerate(t["heads"]):
        cx = X[i] + COLW / 2
        o.append(vline(cx, 72, 96))
        o.append(f'<rect x="{X[i]}" y="96" width="{COLW}" height="44" rx="6" style="fill:var(--fg)"/>')
        o.append(f'<text x="{cx}" y="114" text-anchor="middle" font-size="12.5" font-weight="700" style="fill:var(--bg)">{E(h1)}</text>')
        o.append(f'<text x="{cx}" y="131" text-anchor="middle" font-size="10.5" style="fill:var(--bg)">{E(h2)}</text>')
    tag = t["tag"]
    # 발생 (A): 1~5행
    cx = X[0] + COLW / 2
    prev = 140
    for r, (k, a, b, c) in enumerate(t["A"], start=1):
        o.append(vline(cx, prev, ry(r)))
        o.append(box(X[0], ry(r), k, a, b, c, tag[k]))
        prev = ry(r) + BH
    o.append(vline(cx, prev, ry(7)))
    # 취약성 (B): 1, 3~6행
    cx = X[1] + COLW / 2
    prev = 140
    for r in sorted(t["B"]):
        k, a, b, c = t["B"][r]
        o.append(vline(cx, prev, ry(r)))
        o.append(box(X[1], ry(r), k, a, b, c, tag[k]))
        prev = ry(r) + BH
    o.append(vline(cx, prev, ry(7)))
    # 피해도 (C): 1~6행
    cx = X[2] + COLW / 2
    prev = 140
    for r, (k, a, b, c) in enumerate(t["C"], start=1):
        o.append(vline(cx, prev, ry(r)))
        o.append(box(X[2], ry(r), k, a, b, c, tag[k]))
        prev = ry(r) + BH
    o.append(vline(cx, prev, ry(7)))
    # 갈래 사이: SHAP 순위 → 적응력 후보 (A3 → B3), 변수 역할 → 사전 지정 모형 (B4 → A4)
    y3 = ry(3) + BH / 2
    o.append(f'<line x1="{X[0] + COLW}" y1="{y3}" x2="{X[1] - 2}" y2="{y3}" style="stroke:var(--accent)" stroke-width="1.5" marker-end="url(#pa-a)"/>')
    for dy, key in [(-6, "x_shap"), (13, "x_shap2")]:
        o.append(f'<text x="{(X[0] + COLW + X[1]) / 2}" y="{y3 + dy}" text-anchor="middle" font-size="10" style="fill:var(--accent)" font-weight="600">{E(t[key])}</text>')
    y4 = ry(4) + BH / 2
    o.append(f'<line x1="{X[1]}" y1="{y4}" x2="{X[0] + COLW + 2}" y2="{y4}" stroke="currentColor" stroke-width="1.2" marker-end="url(#pa)"/>')
    o.append(f'<text x="{(X[0] + COLW + X[1]) / 2}" y="{y4 - 6}" text-anchor="middle" font-size="10">{E(t["x_eas"])}</text>')
    o.append(f'<text x="{(X[0] + COLW + X[1]) / 2}" y="{y4 + 13}" text-anchor="middle" font-size="10">{E(t["x_eas2"])}</text>')
    # 결과
    for i, (a, b, c) in enumerate(t["outs"]):
        y = ry(7)
        o.append(f'<rect x="{X[i]}" y="{y}" width="{COLW}" height="{BH}" rx="{BH / 2}" style="fill:var(--surface);stroke:var(--fg)" stroke-width="1.5"/>')
        o.append(f'<text x="{X[i] + COLW / 2}" y="{y + 20}" text-anchor="middle" font-size="12.5" font-weight="700">{E(a)}</text>')
        o.append(f'<text x="{X[i] + COLW / 2}" y="{y + 35}" text-anchor="middle" font-size="11" style="fill:var(--muted)">{E(b)}</text>')
        o.append(f'<text x="{X[i] + COLW / 2}" y="{y + 49}" text-anchor="middle" font-size="11" style="fill:var(--muted)">{E(c)}</text>')
        o.append(vline(X[i] + COLW / 2, y + BH, concl_y))
    o.append(f'<rect x="20" y="{concl_y}" width="{W - 40}" height="44" rx="6" style="fill:var(--fg)"/>')
    o.append(f'<text x="{W / 2}" y="{concl_y + 27}" text-anchor="middle" font-size="12.5" font-weight="700" style="fill:var(--bg)">{E(t["concl"])}</text>')
    # 범례
    for lx, (k, lab) in zip(X, t["legend"]):
        o.append(f'<rect x="{lx}" y="{leg_y - 11}" width="22" height="14" rx="3" {STY[k]}/>')
        o.append(f'<text x="{lx + 30}" y="{leg_y}" font-size="11">{E(lab)}</text>')
    o.append("</svg>")
    return "".join(o)


def figure(lang):
    cap = {"kr": "보고서에 쓰인 분석의 흐름. 초록색 상자가 ML이 기여한 단계로, 발생 기준을 고르고, 변수를 선택하고 모델을 비교·검증하며(nested CV), "
                 "SHAP으로 중요 변수와 방향을 찾고, 오차가 큰 농가를 진단했습니다. 효과의 크기(오즈비, 피해도 배수)와 재배방법별 발생확률 감소량은 "
                 "결과와 무관하게 미리 정한 통계 모형으로 추정했습니다. 두 갈래를 잇는 화살표는 SHAP 순위가 적응력 후보를 제안하고, 취약성 틀이 "
                 "사전 지정 모형의 변수를 정한 경로입니다.",
           "en": "Flow of the analyses in this report. Green boxes are the steps where ML contributed: choosing the occurrence threshold, "
                 "selecting variables and comparing and validating models (nested CV), finding key variables and their direction with SHAP, and "
                 "diagnosing farms with large errors. Effect sizes (odds ratios, severity multipliers) and the risk reduction per practice were "
                 "estimated with statistical models fixed in advance. The arrows between branches show SHAP ranks proposing adaptive-capacity "
                 "candidates and the vulnerability framework fixing the variables of the pre-specified model."}[lang]
    return f'<figure class="pipe">\n    {svg(lang)}\n    <figcaption>{E(cap)}</figcaption>\n  </figure>'


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for lang in ["kr", "en"]:
        open(os.path.join(here, f"pipeline_{lang}.svg.html"), "w").write(figure(lang))
        print("wrote", lang)
