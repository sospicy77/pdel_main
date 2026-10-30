"""Vulnerability scenarios by disease last year → line chart image (PNG + SVG).

Two panels (farms with / without anthracnose last year) on one shared y-axis; x = exposure 0.1 / 0.5 / 0.9.
Current practice = neutral baseline with its 95% band; three improved-practice scenarios in categorical slots 1–3.
Values are the g-computation results of the vulnerability analysis (report Section 06).

Usage: python scenario_chart.py <output prefix>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = [0.1, 0.5, 0.9]
DATA = {
    "Farms with anthracnose last year": {
        "Current practice": ([24.7, 39.8, 56.6], [(17.7, 35.9), (27.0, 52.0), (35.2, 73.2)]),
        "Spray ≤ 2 days before rain": [10.4, 19.3, 32.3],
        "Use weed mat": [18.7, 31.8, 47.7],
        "Both at best": [7.5, 14.3, 24.8],
    },
    "Farms without anthracnose last year": {
        "Current practice": ([22.0, 23.7, 24.3], [(12.8, 30.4), (14.5, 35.2), (15.7, 40.9)]),
        "Spray ≤ 2 days before rain": [9.0, 9.9, 10.2],
        "Use weed mat": [16.5, 17.9, 18.4],
        "Both at best": [6.5, 7.2, 7.4],
    },
}
# reference palette, light mode: neutral baseline + categorical slots 1–3 (validated)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
STYLE = {
    "Current practice": dict(color=INK2, marker="o"),
    "Spray ≤ 2 days before rain": dict(color="#2a78d6", marker="s"),
    "Use weed mat": dict(color="#eb6834", marker="^"),
    "Both at best": dict(color="#1baf7a", marker="D"),
}


def main(prefix):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True, facecolor=SURF)
    for ax, (panel, series) in zip(axes, DATA.items()):
        ax.set_facecolor(SURF)
        cur = series["Current practice"][0]
        labels = []
        for name, vals in series.items():
            st = STYLE[name]
            if name == "Current practice":
                vals, ci = vals
                ax.fill_between(E, [c[0] for c in ci], [c[1] for c in ci], color=INK2, alpha=0.10, lw=0)
            ax.plot(E, vals, color=st["color"], lw=2.4 if name == "Current practice" else 2, zorder=3,
                    marker=st["marker"], ms=8, mec=SURF, mew=2)
            txt = f"{name}  {vals[-1]:.1f}%" if name == "Current practice" else f"{name}  {vals[-1] - cur[-1]:+.1f} pp".replace("-", "−")
            labels.append([vals[-1], txt, name])
        # direct labels at the right end, spread so none sit closer than GAP
        labels.sort(key=lambda r: r[0])
        GAP = 4.6
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + GAP)
        for ly, txt, name in labels:
            ax.annotate(txt, (E[-1], series[name][0][-1] if name == "Current practice" else series[name][-1]),
                        xytext=(1.0, ly), textcoords=("data", "data"), va="center", fontsize=11.5, color=INK,
                        fontweight="bold" if name in ("Current practice", "Both at best") else "normal",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=6))
        ax.set_title(panel, loc="left", fontsize=14, fontweight="bold", color=INK, pad=12)
        ax.set_xticks(E)
        ax.set_xticklabels(["0.1\n(low)", "0.5", "0.9\n(high)"])
        ax.set_xlim(0.04, 1.80)
        ax.set_xlabel("Exposure", fontsize=12)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(length=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(0, 80)
    axes[0].set_yticks(range(0, 81, 20))
    axes[0].set_yticklabels([f"{v}%" for v in range(0, 81, 20)])
    axes[0].set_ylabel("Probability of occurrence (severity > 3%)", fontsize=12)
    fig.suptitle("Practices cut occurrence risk most where exposure is high and disease was present last year",
                 x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.995)
    fig.text(0.02, 0.905, "Mean predicted probability across 141 farms with exposure fixed at each level (g-computation).\n"
             "Labels give the change from current practice at exposure 0.9. Shaded band: 95% interval for current practice.",
             fontsize=11, color=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s["color"], marker=s["marker"], lw=2, ms=7, mec=SURF) for s in STYLE.values()]
    fig.legend(handles, list(STYLE), loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=4, frameon=False,
               fontsize=11, handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.subplots_adjust(wspace=0.06)
    for ext in ("png", "svg"):
        fig.savefig(f"{prefix}.{ext}", dpi=300, facecolor=SURF)
    print("wrote", prefix)


if __name__ == "__main__":
    main(sys.argv[1])
