"""
RF without GIBSIM_pred and Alternaria; focus on Epicoccum
==========================================================
Inputs : Cladosporium, Epicoccum, Hannaella, Papiliotrema, Periconia
Target : FHB_incidence

  1. Tuning + performance  : grid search (repeated CV), 10x 5-fold CV R2/RMSE,
                             comparison with null / linear / RF-without-Epicoccum
  2. Global importance     : out-of-fold permutation, drop-column (paired CV difference)
  3. Epicoccum effect      : SHAP dependence, PDP + ICE, ALE (robust to correlated inputs),
                             direction statistics
  4. Epicoccum interactions: SHAP interaction values, Friedman H^2 with bootstrap,
                             2D PDP with the strongest partners, conditional SHAP by partner level

Run:  python3 Phytobiome/rf_epicoccum_focus.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "epicoccum_focus")
os.makedirs(OUT, exist_ok=True)
SEED = 42
TARGET = "FHB_incidence"
E = "Epicoccum"

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
X = df.drop(columns=[TARGET, "GIBSIM_pred", "Alternaria"])
y = df[TARGET].values
n = len(y)
FEATS = list(X.columns)
OTHERS = [f for f in FEATS if f != E]
COL = dict(zip(FEATS, ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
BLUE, INK, GREY = "#2a78d6", "#52514e", "#d9d9d9"

report = []
def log(s=""):
    print(s)
    report.append(str(s))

log(f"Features: {FEATS}")
log("Spearman correlation with Epicoccum: " +
    ", ".join(f"{f} {df[[E, f]].corr('spearman').iloc[0, 1]:.2f}"
              for f in OTHERS + ["Alternaria", "GIBSIM_pred", TARGET]))

# ------------------------------------------------------------ 1. tuning + performance
log("\n## 1. Tuning and performance")
gs = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                  {"max_features": [1, 2, 3, 5], "min_samples_leaf": [1, 3, 5, 8]},
                  cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED),
                  scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)
best = gs.best_params_
log(f"best params: {best}")
rf_base = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **best)


def oof(model, Xm, seed):
    p = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xm):
        p[te] = clone(model).fit(Xm.iloc[tr], y[tr]).predict(Xm.iloc[te])
    return p


def cv_scores(model, Xm, n_rep=10):
    r2 = np.array([r2_score(y, oof(model, Xm, r)) for r in range(n_rep)])
    return r2


models = {
    "Null (mean)": (DummyRegressor(), X),
    "Linear: 5 fungi": (LinearRegression(), X),
    "Linear: Epicoccum only": (LinearRegression(), X[[E]]),
    "RF: Epicoccum only": (clone(rf_base).set_params(max_features=1), X[[E]]),
    "RF: 4 fungi (no Epicoccum)": (clone(rf_base).set_params(max_features=min(best["max_features"], 4)),
                                   X[OTHERS]),
    "RF: 5 fungi": (rf_base, X),
}
perf, r2_store = [], {}
for name, (m, Xm) in models.items():
    r2 = cv_scores(m, Xm)
    r2_store[name] = r2
    p = oof(m, Xm, 0)
    perf.append(dict(model=name, CV_R2=r2.mean(), CV_R2_sd=r2.std(),
                     RMSE=np.sqrt(((y - p) ** 2).mean()), MAE=np.abs(y - p).mean()))
perf = pd.DataFrame(perf)
log(perf.round(3).to_string(index=False))
perf.round(4).to_csv(os.path.join(OUT, "01_performance.csv"), index=False)
d = r2_store["RF: 5 fungi"] - r2_store["RF: 4 fungi (no Epicoccum)"]
log(f"Gain from adding Epicoccum (paired over 10 CV repeats): dR2 = {d.mean():.3f} "
    f"(min {d.min():.3f}, max {d.max():.3f}, positive in {(d > 0).sum()}/10 repeats)")

rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=SEED,
                           n_jobs=-1, **best).fit(X, y)
log(f"OOB R2 = {rf.oob_score_:.3f}; training R2 = {r2_score(y, rf.predict(X)):.3f}")
p0 = np.mean([oof(rf_base, X, r) for r in range(5)], axis=0)
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.scatter(y, p0, s=26, color=BLUE, edgecolor="white")
ax.plot([0, 100], [0, 100], ls="--", color=INK, lw=.8)
ax.set_xlabel("Observed FHB incidence (%)"); ax.set_ylabel("CV-predicted (%)")
ax.set_title(f"RF (5 fungi, no GIBSIM/Alternaria)\nCV R² = {r2_store['RF: 5 fungi'].mean():.2f}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_obs_vs_pred.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------ 2. importance
log("\n## 2. Global importance")
perm_rows = []
for r in range(5):
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(X)):
        m = clone(rf_base).fit(X.iloc[tr], y[tr])
        perm_rows.append(permutation_importance(m, X.iloc[te], y[te], n_repeats=20,
                                                random_state=r * 10 + k, scoring="r2").importances_mean)
perm = pd.DataFrame(perm_rows, columns=FEATS)
full = r2_store["RF: 5 fungi"]
drop = {}
for f in FEATS:
    m = clone(rf_base).set_params(max_features=min(best["max_features"], 4))
    drop[f] = (full - cv_scores(m, X.drop(columns=f))).mean()
imp = pd.DataFrame({"MDI": rf.feature_importances_, "Perm_dR2": perm.mean(), "Perm_sd": perm.std(),
                    "DropColumn_dR2": pd.Series(drop)}, index=FEATS).sort_values("Perm_dR2", ascending=False)

expl = shap.TreeExplainer(rf)
sv = expl.shap_values(X)
siv = expl.shap_interaction_values(X)
BASE = float(np.ravel(expl.expected_value)[0])
S = pd.DataFrame(sv, columns=FEATS)
imp["mean_abs_SHAP"] = S.abs().mean()
imp["rho(value,SHAP)"] = [X[f].corr(S[f], method="spearman") for f in imp.index]
imp["rank_SHAP"] = imp.mean_abs_SHAP.rank(ascending=False).astype(int)
log(imp.round(3).to_string())
imp.round(4).to_csv(os.path.join(OUT, "02_importance.csv"))

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
order = imp.index[::-1]
axes[0].boxplot([perm[f] for f in order], vert=False, tick_labels=order)
axes[0].axvline(0, color=INK, ls="--", lw=.8); axes[0].set_title("OOF permutation (ΔR²)")
axes[1].barh(order, imp.loc[order, "DropColumn_dR2"], color=[COL[f] for f in order], height=.6)
axes[1].axvline(0, color=INK, lw=.8); axes[1].set_title("Drop-column (Δ CV R²)")
axes[2].barh(order, imp.loc[order, "mean_abs_SHAP"], color=[COL[f] for f in order], height=.6)
axes[2].set_title("mean |SHAP| (%p)")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_importance.png"), dpi=200); plt.close(fig)

plt.figure(); shap.summary_plot(sv, X, show=False); plt.title("SHAP beeswarm (5 fungi)")
plt.savefig(os.path.join(OUT, "02b_shap_beeswarm.png"), dpi=200, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------ 3. Epicoccum effect
log("\n## 3. Epicoccum main effect")
ej = FEATS.index(E)


def ale_1d(model, Xd, f, bins=10):
    """First-order accumulated local effects (Apley & Zhu), centred."""
    z = np.unique(np.quantile(Xd[f], np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(Xd[f], z[1:-1]), 0, len(z) - 2)
    eff = np.zeros(len(z) - 1)
    cnt = np.zeros(len(z) - 1)
    for k in range(len(z) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        lo, hi = Xd[m].copy(), Xd[m].copy()
        lo[f], hi[f] = z[k], z[k + 1]
        eff[k] = (model.predict(hi) - model.predict(lo)).mean()
        cnt[k] = m.sum()
    acc = np.r_[0, np.cumsum(eff)]
    centre = ((acc[:-1] + acc[1:]) / 2 * cnt).sum() / cnt.sum()
    return z, acc - centre


z, ale = ale_1d(rf, X, E)
grid = np.linspace(X[E].min(), X[E].max(), 40)
pdp = np.array([rf.predict(X.assign(**{E: g})).mean() for g in grid])
log(f"ALE of Epicoccum: range {ale.min():.1f} to {ale.max():.1f} %p "
    f"(net change low->high = {ale[-1] - ale[0]:+.1f} %p)")
log(f"PDP of Epicoccum: range {pdp.min():.1f}-{pdp.max():.1f} % (span {pdp.max() - pdp.min():.1f} %p)")
log(f"SHAP of Epicoccum: mean|SHAP| = {S[E].abs().mean():.2f} %p, range {S[E].min():.1f} to {S[E].max():.1f}")
log(f"Spearman(Epicoccum, SHAP_Epicoccum) = {X[E].corr(S[E], method='spearman'):.2f}; "
    f"marginal Spearman(Epicoccum, FHB) = {df[[E, TARGET]].corr('spearman').iloc[0, 1]:.2f}")
qs = pd.qcut(X[E], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
tab = pd.DataFrame({"Epicoccum range": X[E].groupby(qs, observed=True).agg(lambda v: f"{v.min():.1f}-{v.max():.1f}"),
                    "mean SHAP_Epicoccum": S[E].groupby(qs, observed=True).mean(),
                    "observed FHB mean": pd.Series(y).groupby(qs.values, observed=True).mean()})
log(tab.round(2).to_string())
tab.to_csv(os.path.join(OUT, "03_epicoccum_by_quartile.csv"))
pd.DataFrame({"z": z, "ALE": ale}).round(3).to_csv(os.path.join(OUT, "03_epicoccum_ALE.csv"), index=False)

partner = pd.Series(np.abs(siv[:, ej, :]).mean(0), index=FEATS).drop(E).idxmax()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
scat = ax.scatter(X[E], S[E], c=X[partner], cmap="Blues", edgecolor=INK, linewidth=.3, s=30)
fig.colorbar(scat, ax=ax, label=partner)
ax.axhline(0, color=INK, ls="--", lw=.8)
ax.set_xlabel(E); ax.set_ylabel("SHAP value of Epicoccum (%p)")
ax.set_title(f"SHAP dependence (colour = {partner})")
ax = axes[1]
PartialDependenceDisplay.from_estimator(rf, X, [E], kind="both", subsample=98, random_state=SEED,
                                        ax=ax, ice_lines_kw={"alpha": .15, "color": GREY},
                                        pd_line_kw={"color": BLUE, "lw": 2.5})
ax.set_title("PDP (blue) + ICE (grey)")
ax = axes[2]
ax.plot(z, ale, "-o", color=BLUE, lw=2, ms=4)
ax.axhline(0, color=INK, ls="--", lw=.8)
ax.plot(X[E], np.full(n, ale.min() - .5), "|", color=INK, ms=8)
ax.set_xlabel(E); ax.set_ylabel("ALE (%p)")
ax.set_title("ALE (corrects for correlated predictors)")
for a in fig.axes:
    if a.get_label() != "<colorbar>":
        a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_epicoccum_effect.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------ 4. Epicoccum interactions
log("\n## 4. Epicoccum interactions")
inter = pd.DataFrame(np.abs(siv).mean(0), index=FEATS, columns=FEATS)
inter.round(3).to_csv(os.path.join(OUT, "04_shap_interaction_matrix.csv"))
e_int = pd.DataFrame({"mean|SHAP int| (x2)": 2 * inter.loc[E, OTHERS],
                      "share of Epicoccum SHAP mass %": 100 * 2 * inter.loc[E, OTHERS] /
                      (inter.loc[E, E] + 2 * inter.loc[E, OTHERS].sum())})
log(f"Epicoccum main-effect mass = {inter.loc[E, E]:.3f}; interaction mass = "
    f"{2 * inter.loc[E, OTHERS].sum():.3f}")


def pd_at_data(model, Xd, feats):
    rows = np.repeat(Xd.values[None, :, :], len(Xd), axis=0)
    idx = [Xd.columns.get_loc(f) for f in feats]
    rows[:, :, idx] = Xd.values[:, None, idx]
    pred = model.predict(pd.DataFrame(rows.reshape(-1, Xd.shape[1]), columns=Xd.columns))
    v = pred.reshape(len(Xd), len(Xd)).mean(1)
    return v - v.mean()


def h2_all(model):
    pe = pd_at_data(model, X, [E])
    out = {}
    for f in OTHERS:
        pj = pd_at_data(model, X, [E, f])
        pf = pd_at_data(model, X, [f])
        out[f] = ((pj - pe - pf) ** 2).sum() / (pj ** 2).sum()
    # overall H of Epicoccum vs all others
    pr = pd_at_data(model, X, OTHERS)
    fx = model.predict(X); fx = fx - fx.mean()
    out["(all others)"] = ((fx - pe - pr) ** 2).sum() / (fx ** 2).sum()
    return out


h_point = h2_all(rf)
rng = np.random.default_rng(SEED)
hb = []
for b in range(30):
    ib = rng.integers(0, n, n)
    hb.append(h2_all(clone(rf_base).set_params(random_state=b).fit(X.iloc[ib], y[ib])))
hb = pd.DataFrame(hb)
H = pd.DataFrame({"H2": pd.Series(h_point), "lo": hb.quantile(.025), "hi": hb.quantile(.975)})
e_int = e_int.join(H)
log("Friedman H2, Epicoccum vs each partner and vs all others combined:")
log(H.round(3).to_string())
H.round(4).to_csv(os.path.join(OUT, "04_H_statistic.csv"))
log(e_int.round(3).to_string())
e_int.round(4).to_csv(os.path.join(OUT, "04_epicoccum_interactions.csv"))

# direction of each interaction: Epicoccum SHAP slope at low vs high partner
rows = []
for f in OTHERS:
    hi = X[f] >= X[f].median()
    sl = {}
    for lab, m in [("partner low", ~hi), ("partner high", hi)]:
        sl[lab] = np.polyfit(X[E][m], S[E][m], 1)[0]
    pj = 2 * siv[:, ej, FEATS.index(f)]
    rows.append(dict(partner=f, slope_SHAP_Epi_partner_low=sl["partner low"],
                     slope_SHAP_Epi_partner_high=sl["partner high"],
                     rho_product_vs_interaction=pd.Series((X[E] - X[E].mean()) * (X[f] - X[f].mean()))
                     .corr(pd.Series(pj, index=X.index), method="spearman")))
dirtab = pd.DataFrame(rows).set_index("partner")
log("\nSlope of Epicoccum's SHAP vs Epicoccum abundance (%p per unit), split by partner median:")
log(dirtab.round(2).to_string())
dirtab.round(3).to_csv(os.path.join(OUT, "04_interaction_direction.csv"))

top2 = e_int["mean|SHAP int| (x2)"].sort_values(ascending=False).index[:2].tolist()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
PartialDependenceDisplay.from_estimator(rf, X, [(E, top2[0]), (E, top2[1])], ax=axes,
                                        contour_kw={"cmap": "Blues"})
for ax, f in zip(fig.axes[-2:], top2):
    ax.scatter(X[E], X[f], s=10, color=INK, alpha=.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_pdp2d_epicoccum.png"), dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, len(OTHERS), figsize=(16, 3.8), sharey=True)
for ax, f in zip(axes, OTHERS):
    hi = X[f] >= X[f].median()
    for m, c, lab in [(~hi, "#9ec5f0", f"{f} low"), (hi, "#0d3b75", f"{f} high")]:
        ax.scatter(X[E][m], S[E][m], s=22, color=c, edgecolor="white", label=lab)
        b1, b0 = np.polyfit(X[E][m], S[E][m], 1)
        xx = np.linspace(X[E][m].min(), X[E][m].max(), 10)
        ax.plot(xx, b0 + b1 * xx, color=c, lw=2)
    ax.axhline(0, color=INK, ls="--", lw=.6)
    ax.set_title(f"by {f}\n(H² = {H.loc[f, 'H2']:.3f})", fontsize=9)
    ax.set_xlabel(E); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("SHAP value of Epicoccum (%p)")
fig.suptitle("Epicoccum effect conditional on each partner (median split)", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_conditional_shap.png"), dpi=200,
                                bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(inter, cmap="Blues")
ax.set_xticks(range(len(FEATS)), FEATS, rotation=40, ha="right"); ax.set_yticks(range(len(FEATS)), FEATS)
for i in range(len(FEATS)):
    for j in range(len(FEATS)):
        v = inter.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > inter.values.max() / 2 else "#0b0b0b")
fig.colorbar(im, label="mean |SHAP interaction| (%p)")
ax.set_title("SHAP interaction matrix (diagonal = main effect)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_shap_interaction_matrix.png"), dpi=200); plt.close(fig)

with open(os.path.join(OUT, "epicoccum_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
