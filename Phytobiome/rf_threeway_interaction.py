"""
Three-way interaction: Alternaria x Papiliotrema x GIBSIM_pred -> FHB_incidence
================================================================================
Uses the tuned RF from rf_interpretable_ml.py (max_features=3, min_samples_leaf=1).

  1. Friedman's H-statistic   : 2-way H^2 for every pair + 3-way H^2 for the triple,
                                with bootstrap (refit) 95% intervals
  2. Functional (PD) ANOVA    : variance of the joint PD split into main / 2-way / 3-way
                                parts, and the per-sample pure 3-way term
  3. Conditional 2D PDP       : Alternaria x GIBSIM_pred surface at low / mid / high
                                Papiliotrema (small multiples, observed points overlaid)
  4. Conditional SHAP         : SHAP of Alternaria vs Alternaria, faceted by
                                GIBSIM_pred (low/high) x Papiliotrema (low/high)
  5. Surrogate tree           : depth-3 tree on RF predictions using only the 3 variables
                                -> human-readable rules for the joint effect

Run:  python3 Phytobiome/rf_threeway_interaction.py
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "threeway")
os.makedirs(OUT, exist_ok=True)
SEED = 42
TARGET = "FHB_incidence"
A, P, G = "Alternaria", "Papiliotrema", "GIBSIM_pred"
TRIO = [A, P, G]
RF_PARAMS = dict(max_features=3, min_samples_leaf=1, max_depth=None)

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
X = df.drop(columns=TARGET)
y = df[TARGET].values
n = len(X)

report = []
def log(s=""):
    print(s)
    report.append(str(s))

rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, **RF_PARAMS).fit(X, y)


# ------------------------------------------------------------ PD machinery
def pd_at_data(model, Xd, feats):
    """Centered partial dependence of `feats`, evaluated at each observed x_feats(i)."""
    rows = np.repeat(Xd.values[None, :, :], len(Xd), axis=0)       # (i, j, p)
    idx = [Xd.columns.get_loc(f) for f in feats]
    rows[:, :, idx] = Xd.values[:, None, idx]                       # x_S from i, x_-S from j
    pred = model.predict(pd.DataFrame(rows.reshape(-1, Xd.shape[1]), columns=Xd.columns)) \
        .reshape(len(Xd), len(Xd))
    pdv = pred.mean(1)
    return pdv - pdv.mean()


def decompose(model, Xd):
    """Functional-ANOVA decomposition of the joint PD of TRIO at the data points."""
    pdf = {s: pd_at_data(model, Xd, list(s))
           for k in (1, 2, 3) for s in itertools.combinations(TRIO, k)}
    main = {f: pdf[(f,)] for f in TRIO}
    two = {}
    for a, b in itertools.combinations(TRIO, 2):
        two[(a, b)] = pdf[(a, b)] - pdf[(a,)] - pdf[(b,)]
    joint = pdf[tuple(TRIO)]
    three = joint - sum(two.values()) - sum(main.values())
    H2 = {f"{a} x {b}": (two[(a, b)] ** 2).sum() / (pdf[(a, b)] ** 2).sum()
          for a, b in two}
    H2[f"{A} x {P} x {G}"] = (three ** 2).sum() / (joint ** 2).sum()
    var = {"main: " + f: v.var() for f, v in main.items()}
    var.update({"2-way: " + " x ".join(k): v.var() for k, v in two.items()})
    var["3-way: A x P x G"] = three.var()
    return H2, var, joint, three, two


# ------------------------------------------------------------ 1-2. H-statistic + ANOVA
log("## 1. Friedman H^2 (share of joint PD variance not explained by lower-order terms)")
H2, var, joint, three, two = decompose(rf, X)

rng = np.random.default_rng(SEED)
boot = []
for b in range(30):
    ib = rng.integers(0, n, n)
    m = RandomForestRegressor(n_estimators=300, random_state=b, n_jobs=-1, **RF_PARAMS) \
        .fit(X.iloc[ib], y[ib])
    boot.append(decompose(m, X)[0])
boot = pd.DataFrame(boot)
Htab = pd.DataFrame({"H2": pd.Series(H2), "boot_2.5%": boot.quantile(.025),
                     "boot_97.5%": boot.quantile(.975)})
log(Htab.round(3).to_string())
Htab.to_csv(os.path.join(OUT, "H_statistic.csv"))

log("\n## 2. Variance decomposition of the joint PD f(A, P, G)")
var = pd.Series(var)
share = (var / var.sum()).sort_values(ascending=False)
log(pd.DataFrame({"variance": var[share.index], "share": share}).round(3).to_string())
share.to_csv(os.path.join(OUT, "pd_variance_decomposition.csv"))
log(f"joint PD range: {joint.min() + y.mean():.1f} - {joint.max() + y.mean():.1f} %")
log(f"pure 3-way term: SD = {three.std():.2f} %p, range {three.min():.1f} to {three.max():.1f} %p")

pd.DataFrame({A: X[A], P: X[P], G: X[G], "FHB": y, "joint_PD": joint,
              "pure_3way": three, **{"2way_" + "_".join(k): v for k, v in two.items()}}) \
    .round(3).to_csv(os.path.join(OUT, "per_sample_decomposition.csv"), index=False)

BLUE = "#2a6f97"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
h = Htab.sort_values("H2")
axes[0].barh(h.index, h.H2, color=BLUE, height=.55)
axes[0].errorbar(h.H2, h.index, xerr=[h.H2 - h["boot_2.5%"], h["boot_97.5%"] - h.H2],
                 fmt="none", ecolor="#333", capsize=3, lw=1)
axes[0].set_xlabel("Friedman H²  (bootstrap 95% interval)")
axes[0].set_title("Interaction strength")
s = share[::-1]
axes[1].barh(s.index, s.values * 100,
             color=["#c1121f" if i.startswith("3") else "#8d99ae" if i.startswith("2") else BLUE
                    for i in s.index], height=.55)
for yi, v in enumerate(s.values * 100):
    axes[1].text(v + .5, yi, f"{v:.1f}%", va="center", fontsize=8)
axes[1].set_xlabel("% of variance of joint PD f(A, P, G)")
axes[1].set_title("Functional ANOVA decomposition")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_H_and_decomposition.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------ 3. Conditional 2D PDP
log("\n## 3. Alternaria x GIBSIM_pred PDP at Papiliotrema levels")
ga = np.linspace(X[A].min(), X[A].max(), 40)
gg = np.linspace(X[G].min(), X[G].max(), 40)
p_cut = X[P].quantile([1 / 3, 2 / 3]).values
p_grp = np.digitize(X[P], p_cut)                                   # 0 / 1 / 2 tertile
levels = [X[P][p_grp == k].median() for k in range(3)]
names = ["low", "mid", "high"]
surf = []
for pv in levels:
    AA, GG = np.meshgrid(ga, gg)
    rows = np.repeat(X.values[None, :, :], AA.size, axis=0)        # (grid, sample, p)
    rows[:, :, X.columns.get_loc(A)] = AA.ravel()[:, None]
    rows[:, :, X.columns.get_loc(G)] = GG.ravel()[:, None]
    rows[:, :, X.columns.get_loc(P)] = pv
    Z = rf.predict(pd.DataFrame(rows.reshape(-1, X.shape[1]), columns=X.columns)) \
        .reshape(AA.size, n).mean(1).reshape(AA.shape)
    surf.append(Z)
vmin, vmax = min(z.min() for z in surf), max(z.max() for z in surf)
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, .05]})
for k, (ax, Z) in enumerate(zip(axes[:3], surf)):
    im = ax.contourf(ga, gg, Z, levels=np.linspace(vmin, vmax, 15), cmap="Blues")
    ax.contour(ga, gg, Z, levels=[30, 50, 70], colors="#333", linewidths=.6)
    m = p_grp == k
    ax.scatter(X[A][m], X[G][m], c=y[m], cmap="Blues", vmin=0, vmax=100,
               edgecolor="#c1121f", linewidth=.9, s=34)
    ax.set_title(f"Papiliotrema {names[k]} (≈{levels[k]:.1f}, n={m.sum()})")
    ax.set_xlabel(A); ax.set_ylabel(G if k == 0 else "")
    log(f"Papiliotrema {names[k]:4s} ({levels[k]:.2f}): predicted FHB "
        f"min {Z.min():.1f} / max {Z.max():.1f}; "
        f"high-GIBSIM effect at low Alternaria = {Z[-1, 0] - Z[0, 0]:.1f} %p, "
        f"at high Alternaria = {Z[-1, -1] - Z[0, -1]:.1f} %p")
fig.colorbar(im, cax=axes[3], label="Partial dependence: FHB incidence (%)")
fig.suptitle("Alternaria × GIBSIM_pred surface conditional on Papiliotrema "
             "(dots = observed samples in that tertile, fill = observed FHB)", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_conditional_pdp.png"), dpi=200,
                                bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------ 4. Conditional SHAP
log("\n## 4. SHAP of Alternaria conditioned on GIBSIM_pred x Papiliotrema")
expl = shap.TreeExplainer(rf)
sv = pd.DataFrame(expl.shap_values(X), columns=X.columns)
siv = expl.shap_interaction_values(X)
ia, ip, ig = (X.columns.get_loc(f) for f in TRIO)
trio_sum = sv[TRIO].sum(axis=1)
pd.DataFrame({"SHAP_" + f: sv[f] for f in TRIO} | {
    "SHAP_trio_sum": trio_sum, "SHAPint_A_P": 2 * siv[:, ia, ip],
    "SHAPint_A_G": 2 * siv[:, ia, ig], "SHAPint_P_G": 2 * siv[:, ip, ig]}) \
    .round(3).to_csv(os.path.join(OUT, "shap_trio.csv"), index=False)

g_hi = X[G] >= X[G].median()
p_hi = X[P] >= X[P].median()
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
rows_out = []
for r, gh in enumerate([True, False]):
    for c, ph in enumerate([False, True]):
        ax = axes[r, c]
        m = (g_hi == gh) & (p_hi == ph)
        ax.scatter(X[A], sv[A], s=14, color="#d9d9d9")
        ax.scatter(X[A][m], sv[A][m], s=34, color=BLUE, edgecolor="white")
        ax.axhline(0, color="#999", lw=.6, ls="--")
        ax.set_title(f"GIBSIM {'high' if gh else 'low'} · Papiliotrema {'high' if ph else 'low'}"
                     f"  (n={m.sum()}, mean FHB {y[m].mean():.0f}%)", fontsize=9)
        rows_out.append(dict(GIBSIM="high" if gh else "low", Papiliotrema="high" if ph else "low",
                             n=int(m.sum()), mean_FHB=y[m].mean(),
                             mean_SHAP_Alt_lowAlt=sv[A][m & (X[A] < 7.2)].mean(),
                             mean_SHAP_Alt_highAlt=sv[A][m & (X[A] >= 7.2)].mean()))
for ax in axes[1]:
    ax.set_xlabel(A)
for ax in axes[:, 0]:
    ax.set_ylabel("SHAP value of Alternaria (%p)")
fig.suptitle("Alternaria effect by GIBSIM_pred × Papiliotrema (grey = all samples)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_conditional_shap.png"), dpi=200); plt.close(fig)
cs = pd.DataFrame(rows_out)
log(cs.round(2).to_string(index=False))
cs.to_csv(os.path.join(OUT, "conditional_shap_summary.csv"), index=False)

# ------------------------------------------------------------ 5. Surrogate tree
log("\n## 5. Surrogate tree (depth 3) on RF predictions, using only the trio")
yhat = rf.predict(X)
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=6, random_state=SEED).fit(X[TRIO], yhat)
log(f"fidelity to RF (R2 vs RF prediction) = {r2_score(yhat, tree.predict(X[TRIO])):.3f}; "
    f"R2 vs observed = {r2_score(y, tree.predict(X[TRIO])):.3f}")
rules = export_text(tree, feature_names=TRIO, decimals=2)
log(rules)
fig, ax = plt.subplots(figsize=(16, 6.5))
plot_tree(tree, feature_names=TRIO, filled=True, rounded=True, impurity=False,
          precision=1, fontsize=9, ax=ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_surrogate_tree.png"), dpi=200); plt.close(fig)

with open(os.path.join(OUT, "threeway_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
