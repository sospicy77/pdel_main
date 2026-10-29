"""Bar chart of genus contributions from the asymmetric Shapley analysis
(flowering temperature + RH; values from output/causal_shapley_weather_TRH):
  - without weather adjustment : genera credited before weather (ASV genera-first)
  - weather-adjusted ("pure")  : weather credited before genera (ASV env-first)"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "output", "causal_shapley_weather_TRH")
OUT = os.path.join(SRC, "asv_pure_contribution_bar.png")

genera = ["Alternaria", "Hannaella", "Epicoccum", "Periconia"]
tab = pd.read_csv(os.path.join(SRC, "mean_abs_attribution.csv"), index_col=0)
series = [("Without weather adjustment", tab.loc[genera, "ASV genera-first"].round(1).values, "#a3a29c"),
          ("Weather-adjusted (pure)", tab.loc[genera, "ASV env-first"].round(1).values, "#2a78d6")]

x = np.arange(len(genera))
w = 0.36
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, (label, vals, col) in enumerate(series):
    bars = ax.bar(x + (k - 0.5) * w, vals, width=w * 0.94, color=col, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color="#0b0b0b")
ax.set_xticks(x, genera)
for lab in ax.get_xticklabels():
    lab.set_fontstyle("italic")
ax.set_ylabel("Contribution to FHB incidence\n(Asymmetric Shapley, %p)")
ax.set_ylim(0, max(series[0][1].max(), series[1][1].max()) * 1.15)
ax.yaxis.grid(True, color="#e4e3df", lw=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(OUT)
