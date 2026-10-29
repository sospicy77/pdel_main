"""Bar chart of pure (weather-adjusted) contributions from the asymmetric Shapley analysis
(flowering temperature + RH credited first; values from output/causal_shapley_weather_TRH)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "causal_shapley_weather_TRH", "asv_pure_contribution_bar.png")

genera = ["Alternaria", "Hannaella", "Epicoccum", "Periconia"]
values = [2.8, 1.2, 1.0, 1.2]          # mean |ASV| (FHB %p), environment-first ordering

fig, ax = plt.subplots(figsize=(5.2, 4))
bars = ax.bar(genera, values, width=0.55, color="#2a78d6")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.1f}", ha="center", va="bottom",
            fontsize=10, color="#0b0b0b")
ax.set_ylabel("Pure contribution to FHB incidence\n(Asymmetric Shapley, %p)")
ax.set_ylim(0, max(values) * 1.2)
ax.yaxis.grid(True, color="#e4e3df", lw=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
for lab in ax.get_xticklabels():
    lab.set_fontstyle("italic")
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(OUT)
