"""Flow chart for running the vulnerability assessment alone (occurrence modelling → variable selection →
vulnerability model → use). Each step is tagged ML / Stats / Design. Titles only, large type.

Usage: python vuln_flow_diagram.py <output html>   (standalone page in the report's style)
"""
import os
import sys
from html import escape as E

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_diagram import STY, TAGF  # noqa: E402

W, BX, BW, BH, PITCH, Y0 = 720, 190, 400, 46, 76, 84

STEPS = [
    # (no, kind, title, label on the arrow to the next step)
    ("1", "ml", "Occurrence definition", "occurrence labels"),
    ("2", "ml", "Occurrence modelling", "predictive signal"),
    ("3", "ml", "Variable importance (SHAP)", "key variables"),
    ("4", "rule", "Role assignment (E · S · A)", "adaptive candidates"),
    ("5", "st", "Adaptive-variable selection", "2 practices"),
    ("6", "st", "Model-structure tests", "model structure"),
    ("7", "st", "Vulnerability model fit", "weights · CIs"),
    ("8", "st", "Scenarios (g-computation)", "risk reduction"),
    ("9", "ml", "Contributions (Shapley)", "priority practices"),
    ("10", "st", "Vulnerability index", None),
]
PHASES = [("Occurrence", "modelling", 1, 3), ("Variable", "selection", 4, 5), ("Vulnerability", "model", 6, 7),
          ("Use", "", 8, 10)]
TAG = {"ml": "ML", "st": "Stats", "rule": "Design"}


def y(i):
    return Y0 + (i - 1) * PITCH


def svg():
    leg_y = y(len(STEPS)) + BH + 108
    H = leg_y + 24
    cx = BX + BW / 2
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="vflow" style="color:var(--fg);font-family:var(--f-body)" '
         'fill="currentColor" aria-label="Vulnerability assessment flow chart. Steps 1 to 3 (occurrence definition, occurrence '
         'modelling, variable importance) are ML, step 4 (role assignment) is a design step, steps 5 to 8 and 10 are statistical, '
         'and step 9 (Shapley contributions) is an ML interpretation method applied to the statistical model.">']
    o.append('<defs><marker id="va" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>')
    o.append(f'<rect x="{BX}" y="14" width="{BW}" height="40" rx="20" style="fill:var(--fg)"/>')
    o.append(f'<text x="{cx}" y="40" text-anchor="middle" font-size="17" font-weight="700" style="fill:var(--bg)">Survey data · 141 farms</text>')
    o.append(f'<line x1="{cx}" y1="54" x2="{cx}" y2="{Y0 - 2}" stroke="currentColor" stroke-width="1.6" marker-end="url(#va)"/>')
    for l1, l2, a, b in PHASES:
        y1, y2 = y(a), y(b) + BH
        o.append(f'<path d="M{BX - 22},{y1} h-10 v{y2 - y1} h10" fill="none" style="stroke:var(--muted)" stroke-width="1.5"/>')
        cy = (y1 + y2) / 2
        if l2:
            o.append(f'<text x="{BX - 42}" y="{cy - 3}" text-anchor="end" font-size="15" font-weight="700" style="fill:var(--muted)">{E(l1)}</text>')
            o.append(f'<text x="{BX - 42}" y="{cy + 16}" text-anchor="end" font-size="15" font-weight="700" style="fill:var(--muted)">{E(l2)}</text>')
        else:
            o.append(f'<text x="{BX - 42}" y="{cy + 5}" text-anchor="end" font-size="15" font-weight="700" style="fill:var(--muted)">{E(l1)}</text>')
    for i, (no, k, title, nxt) in enumerate(STEPS, start=1):
        top = y(i)
        o.append(f'<rect x="{BX}" y="{top}" width="{BW}" height="{BH}" rx="7" {STY[k]}/>')
        o.append(f'<circle cx="{BX + 22}" cy="{top + BH / 2}" r="13" style="{TAGF[k]}"/>')
        o.append(f'<text x="{BX + 22}" y="{top + BH / 2 + 5}" text-anchor="middle" font-size="13" font-weight="700" style="fill:var(--surface)">{no}</text>')
        o.append(f'<text x="{BX + 46}" y="{top + BH / 2 + 6}" font-size="17" font-weight="700">{E(title)}</text>')
        tw = 18 + 8.5 * len(TAG[k])
        o.append(f'<rect x="{BX + BW - tw - 10}" y="{top + BH / 2 - 11}" width="{tw}" height="22" rx="11" style="{TAGF[k]}"/>')
        o.append(f'<text x="{BX + BW - tw / 2 - 10}" y="{top + BH / 2 + 5}" text-anchor="middle" font-size="13" font-weight="700" '
                 f'style="fill:var(--surface)">{TAG[k]}</text>')
        if nxt:
            o.append(f'<line x1="{cx}" y1="{top + BH}" x2="{cx}" y2="{y(i + 1) - 2}" stroke="currentColor" stroke-width="1.6" marker-end="url(#va)"/>')
            o.append(f'<text x="{cx + 14}" y="{top + BH + (PITCH - BH) / 2 + 5}" font-size="13.5" font-weight="600" style="fill:var(--muted)">{E(nxt)}</text>')
    last = y(len(STEPS)) + BH
    o.append(f'<line x1="{cx}" y1="{last}" x2="{cx}" y2="{last + 24}" stroke="currentColor" stroke-width="1.6" marker-end="url(#va)"/>')
    o.append(f'<rect x="{BX}" y="{last + 26}" width="{BW}" height="40" rx="20" style="fill:var(--fg)"/>')
    o.append(f'<text x="{cx}" y="{last + 52}" text-anchor="middle" font-size="17" font-weight="700" style="fill:var(--bg)">Vulnerability by farm · practice effects</text>')
    for lx, k, lab in [(BX - 10, "ml", "ML"), (BX + 130, "st", "Stats"), (BX + 270, "rule", "Design")]:
        o.append(f'<rect x="{lx}" y="{leg_y - 15}" width="30" height="20" rx="4" {STY[k]}/>')
        o.append(f'<text x="{lx + 40}" y="{leg_y}" font-size="15" font-weight="700">{lab}</text>')
    o.append("</svg>")
    return "".join(o)


CAPTION = ("Vulnerability assessment run on its own. ML steps (green) find the predictive signal and the key variables; the "
           "statistical model then selects the adaptive variables and estimates effects, and Shapley (an ML interpretation method) "
           "decomposes its predictions.")


def figure():
    return f'<figure class="vf">\n    {svg()}\n    <figcaption>{E(CAPTION)}</figcaption>\n  </figure>'


if __name__ == "__main__":
    from make_report_v2 import FONTS
    from make_report_v3 import EXTRA_CSS, STYLE
    out = sys.argv[1]
    page = (f'<title>Vulnerability Assessment Flow</title>\n{FONTS}\n{STYLE}\n{EXTRA_CSS}\n'
            "<style>.vf svg { width: 100%; max-width: 760px; min-width: 520px; height: auto; display: block; margin: 0 auto; }</style>\n"
            '<div class="wrap" lang="en"><header><div class="eyebrow">Pepper anthracnose survey · Vulnerability assessment</div>'
            "<h1>Vulnerability assessment flow</h1>"
            '<p class="sub">From occurrence modelling to variable selection, the vulnerability model and its use · ML / Stats / Design per step</p></header>\n'
            f"<section>\n  {figure()}\n</section></div>")
    import re
    page = re.sub(r"/\*[^*]*[\uac00-\ud7a3][^*]*\*/", "", page)   # drop Korean CSS comments
    open(out, "w").write(page)
    print("wrote", out)
