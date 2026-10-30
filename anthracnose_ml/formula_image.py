"""Vulnerability formula as an image (white background, black text): final formula first, then each component,
then a key to the symbols. Coefficients come from results_v4/vulnerability/vulnerability.json.

Usage: python formula_image.py <vulnerability.json> <output prefix>
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, RULE = "#000000", "#000000", "#bbbbbb"


def main(js, prefix):
    r = json.load(open(js))
    p, ref = r["params"], r["practice_ref"]
    w_t, w_m = p["w"]["7~8월_사전경과_최대"], p["w"]["제초매트"]
    th = p["theta"]
    lo, hi = ref["7~8월_사전경과_최대"]["low"], ref["7~8월_사전경과_최대"]["high"]
    plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11, 6.3), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.3)

    ax.text(0.45, 5.85, "Vulnerability formula", fontsize=19, fontweight="bold", color=INK, va="center")
    ax.text(0.45, 5.05, r"$V \;=\; E_{\mathrm{eff}} \times S \times (1 - A)$", fontsize=26, color=INK, va="center")
    ax.plot([0.45, 10.55], [4.45, 4.45], color=RULE, lw=1)

    rows = [
        ("Effective exposure", r"$E_{\mathrm{eff}} \;=\; E^{\,%.2f}\cdot \exp(%.2f \cdot L \cdot E)\;/\;\max$" % (p["alpha"], p["theta_x"])),
        ("Sensitivity", r"$S \;=\; \exp(%.2f\,C + %.2f\,N - %.2f\,Z)\;/\;\max$" % (th["연작"], th["인근고추밭_50m"], -p["beta_size"])),
        ("Adaptive capacity", r"$A \;=\; 1 - (1 - %.2f\,a_{\mathrm{spray}})\,(1 - %.2f\,a_{\mathrm{mat}})$" % (w_t, w_m)),
    ]
    for i, (lab, f) in enumerate(rows):
        y = 3.95 - i * 0.68
        ax.text(0.45, y, lab, fontsize=14, color=INK2, va="center")
        ax.text(3.0, y, f, fontsize=19, color=INK, va="center")
    ax.plot([0.45, 10.55], [1.85, 1.85], color=RULE, lw=1)

    key = [
        (r"$E$", "exposure (0–1)"), (r"$L$", "anthracnose last year (0/1)"),
        (r"$C$", "continuous cropping (0/1)"), (r"$N$", "pepper field within 50 m (0/1)"),
        (r"$Z$", "farm size (log no. of plants, standardized)"),
        (r"$a_{\mathrm{spray}}$", f"spray before rain: ≥{hi:g} days = 0, ≤{lo:g} days = 1"),
        (r"$a_{\mathrm{mat}}$", "weed mat (1 = used)"), (r"$/\,\max$", "scaled to 0–1 by the sample maximum"),
    ]
    for j, (sym, desc) in enumerate(key):
        col, row = j % 2, j // 2
        x, y = 0.45 + col * 5.3, 1.45 - row * 0.36
        ax.text(x, y, sym, fontsize=13.5, color=INK, va="center")
        ax.text(x + 0.85, y, desc, fontsize=12, color=INK2, va="center")
    for ext in ("png", "svg"):
        fig.savefig(f"{prefix}.{ext}", dpi=300, facecolor="white")
    print("wrote", prefix)


if __name__ == "__main__":
    main(*sys.argv[1:3])
