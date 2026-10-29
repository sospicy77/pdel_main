"""취약성 평가만 수행할 때의 흐름도 (발생 모델링 → 변수 선정 → 취약성 모형 → 활용). 단계마다 ML / 통계 / 설계 표시.

사용법: python vuln_flow_diagram.py <출력 html>   (보고서와 같은 스타일의 독립 페이지)
"""
import os
import sys
from html import escape as E

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_diagram import STY, TAGF  # noqa: E402

W, BX, BW, BH, PITCH, Y0 = 860, 150, 420, 60, 92, 70

STEPS = [
    # (단계, 종류, 제목, 설명1, 설명2, 다음 단계로 넘기는 것)
    ("1", "ml", "발생 정의", "피해도 임계값 0–3% × 분류 모델, CV AUC 비교", "가장 잘 구분되는 3%를 발생으로 정의", "발생 라벨 (40 / 101)"),
    ("2", "ml", "발생 모델링", "선택법 4종 × 모델 6종 × 변수 수 k", "변수 선택은 fold 안 · nested CV로 성능 확인", "예측 신호 확인 (AUC ≈ 0.79–0.81)"),
    ("3", "ml", "변수 중요도", "SHAP: 중요 변수와 위험 방향", "안정성 선택: 변수별 근거 강도", "중요 변수 · 방향 · 빈도"),
    ("4", "rule", "역할 지정 (E · S · A)", "E 노출값 · S 작년 발생·연작·인근 고추밭 (+규모 보정)", "A 후보: 농가가 바꿀 수 있는 관행만", "적응력 후보 관행"),
    ("5", "st", "적응력 변수 선정", "후보 조합 9개를 같은 취약성 모형에 넣어 비교", "CV AUC · AIC · 가중치 부트스트랩 구간", "적응력 2개 (강우 전 방제, 제초매트)"),
    ("6", "st", "모형 구조 검정", "노출값 지수 α, 작년 발생 × 노출값 항", "관행 간 시너지 (우도비 검정)", "확정된 모형 구조"),
    ("7", "st", "취약성 모형 적합", "H = λ · Eᵅ · S · (1 − A),  P = 1 − e⁻ᴴ", "A = 1 − Π(1 − wⱼaⱼ) · 부트스트랩 300회", "가중치 w · 모수 · 95% CI"),
    ("8", "st", "시나리오 (g-computation)", "노출값 0.1 / 0.5 / 0.9 × 관행 최선 · 현재", "작년 발생 여부별", "관행별 발생확률 감소 (%p)"),
    ("9", "st", "기여도 분해 (Shapley)", "변수별 발생확률 기여 (%p)", "적응력 안 관행 비중", "우선순위가 높은 관행"),
    ("10", "st", "취약성 지수와 검증", "V = E_eff × S × (1 − A), 농가별 계산", "발생 판별력 (AUC)", None),
]
PHASES = [("발생 모델링", 1, 3), ("변수 선정", 4, 5), ("취약성 모형", 6, 7), ("활용", 8, 10)]
TAG = {"ml": "ML", "st": "통계", "rule": "설계"}


def y(i):
    return Y0 + (i - 1) * PITCH


def svg():
    H = y(len(STEPS)) + BH + 70
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="vflow" style="color:var(--fg);font-family:var(--f-body)" fill="currentColor" '
         f'aria-label="취약성 평가 흐름도. 1 발생 정의, 2 발생 모델링, 3 변수 중요도는 ML 단계이고, 4 역할 지정은 설계 단계, '
         f'5 적응력 변수 선정부터 10 취약성 지수와 검증까지는 통계 단계이다.">']
    o.append('<defs><marker id="va" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>')
    # 시작
    o.append(f'<rect x="{BX}" y="12" width="{BW}" height="36" rx="18" style="fill:var(--fg)"/>')
    o.append(f'<text x="{BX + BW / 2}" y="35" text-anchor="middle" font-size="12.5" font-weight="700" style="fill:var(--bg)">'
             f'설문 자료 · 141개 농가 (관행 · 민감도 · 기상 · 노출값 · 피해도)</text>')
    o.append(f'<line x1="{BX + BW / 2}" y1="48" x2="{BX + BW / 2}" y2="{Y0 - 2}" stroke="currentColor" stroke-width="1.2" marker-end="url(#va)"/>')
    # 단계 구분 (왼쪽 괄호)
    for name, a, b in PHASES:
        y1, y2 = y(a), y(b) + BH
        o.append(f'<path d="M118,{y1} h-8 v{y2 - y1} h8" fill="none" stroke="currentColor" stroke-width="1" style="stroke:var(--muted)"/>')
        cy = (y1 + y2) / 2
        o.append(f'<text x="100" y="{cy + 4}" text-anchor="end" font-size="12" font-weight="600" style="fill:var(--muted)">{E(name)}</text>')
    for i, (no, k, title, s1, s2, nxt) in enumerate(STEPS, start=1):
        top = y(i)
        o.append(f'<rect x="{BX}" y="{top}" width="{BW}" height="{BH}" rx="6" {STY[k]}/>')
        o.append(f'<circle cx="{BX + 18}" cy="{top + 20}" r="10" style="{TAGF[k]}"/>')
        o.append(f'<text x="{BX + 18}" y="{top + 24}" text-anchor="middle" font-size="10.5" font-weight="700" style="fill:var(--surface)">{no}</text>')
        o.append(f'<text x="{BX + 36}" y="{top + 24}" font-size="13" font-weight="700">{E(title)}</text>')
        o.append(f'<text x="{BX + 36}" y="{top + 41}" font-size="11" style="fill:var(--muted)">{E(s1)}</text>')
        o.append(f'<text x="{BX + 36}" y="{top + 55}" font-size="11" style="fill:var(--muted)">{E(s2)}</text>')
        tw = 14 + 8 * len(TAG[k])
        o.append(f'<rect x="{BX + BW - tw - 8}" y="{top + 9}" width="{tw}" height="17" rx="8.5" style="{TAGF[k]}"/>')
        o.append(f'<text x="{BX + BW - tw / 2 - 8}" y="{top + 21.5}" text-anchor="middle" font-size="10.5" font-weight="700" '
                 f'style="fill:var(--surface)">{TAG[k]}</text>')
        if nxt:
            cx = BX + BW / 2
            o.append(f'<line x1="{cx}" y1="{top + BH}" x2="{cx}" y2="{y(i + 1) - 2}" stroke="currentColor" stroke-width="1.2" marker-end="url(#va)"/>')
            o.append(f'<text x="{cx + 12}" y="{top + BH + (PITCH - BH) / 2 + 4}" font-size="11" font-weight="600">{E(nxt)}</text>')
    # 산출
    last = y(len(STEPS)) + BH
    cx = BX + BW / 2
    o.append(f'<line x1="{cx}" y1="{last}" x2="{cx}" y2="{last + 18}" stroke="currentColor" stroke-width="1.2" marker-end="url(#va)"/>')
    o.append(f'<rect x="{BX}" y="{last + 20}" width="{BW}" height="36" rx="18" style="fill:var(--fg)"/>')
    o.append(f'<text x="{cx}" y="{last + 43}" text-anchor="middle" font-size="12.5" font-weight="700" style="fill:var(--bg)">'
             f'산출: 농가별 취약성 · 관행별 감소 효과 · 우선 관행</text>')
    # 오른쪽: 종류 설명
    lx, ly = 620, y(1)
    notes = [("ml", "ML", ["예측 성능으로 판단", "(교차검증 · nested CV)", "변수 후보와 방향 탐색"]),
             ("st", "통계", ["구조를 미리 정한 모형", "모수 · 신뢰구간 · 검정", "효과 크기와 시나리오"]),
             ("rule", "설계", ["분석자가 정하는 규칙", "(변수의 역할 지정)", ""])]
    for j, (k, lab, lines) in enumerate(notes):
        top = ly + j * 96
        o.append(f'<rect x="{lx}" y="{top}" width="220" height="80" rx="6" {STY[k]}/>')
        o.append(f'<text x="{lx + 12}" y="{top + 22}" font-size="12.5" font-weight="700">{lab}</text>')
        for m, ln in enumerate(lines):
            o.append(f'<text x="{lx + 12}" y="{top + 40 + m * 14}" font-size="11" style="fill:var(--muted)">{E(ln)}</text>')
    # ML → 통계 연결 설명
    yy = y(4) + 10
    o.append(f'<text x="{lx}" y="{yy}" font-size="11.5" font-weight="700">ML과 통계의 접점</text>')
    for m, ln in enumerate(["3단계의 SHAP·안정성 결과가 4·5단계", "적응력 후보를 제안하고, 최종 선정과", "효과 크기는 통계 모형(5–10단계)이", "결과와 무관한 기준으로 정합니다."]):
        o.append(f'<text x="{lx}" y="{yy + 18 + m * 15}" font-size="11" style="fill:var(--muted)">{E(ln)}</text>')
    o.append("</svg>")
    return "".join(o)


CAPTION = ("취약성 평가만 수행할 때의 흐름. 1–3단계(ML)는 발생을 예측할 수 있는 신호가 있는지 확인하고 중요 변수와 그 방향을 찾습니다. "
           "4단계에서 변수를 노출(E)·민감도(S)·적응력(A)으로 나누고, 5단계부터는 구조를 미리 정한 통계 모형으로 적응력 변수를 고르고 "
           "효과 크기·시나리오·기여도·취약성 지수를 계산합니다. 화살표 옆 글자는 다음 단계로 넘기는 결과입니다.")


def figure():
    return f'<figure class="vf">\n    {svg()}\n    <figcaption>{E(CAPTION)}</figcaption>\n  </figure>'


if __name__ == "__main__":
    from make_report_v2 import FONTS
    from make_report_v3 import EXTRA_CSS, STYLE
    out = sys.argv[1]
    page = (f"<title>취약성 평가 흐름도</title>\n{FONTS}\n{STYLE}\n{EXTRA_CSS}\n"
            "<style>.vf svg { width: 100%; min-width: 640px; height: auto; display: block; }</style>\n"
            '<div class="wrap"><header><div class="eyebrow">고추탄저병 설문 · 취약성 평가</div>'
            "<h1>취약성 평가 흐름도</h1>"
            '<p class="sub">발생 모델링부터 변수 선정, 취약성 모형, 활용까지 · 단계별 ML / 통계 / 설계 구분</p></header>\n'
            f"<section>\n  {figure()}\n</section></div>")
    open(out, "w").write(page)
    print("wrote", out)
