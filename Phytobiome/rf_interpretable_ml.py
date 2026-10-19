"""
Interpretable ML analysis of FHB incidence with Random Forest
==============================================================
Data   : Phytobiome/using_data.xlsx  (98 samples)
Target : FHB_incidence (%)
Inputs : all other columns (6 fungal genera + GIBSIM_pred)

Steps
  1. EDA                    : distributions, Spearman correlations (collinearity check)
  2. RF tuning              : grid search with repeated K-fold CV
  3. Performance            : OOB / repeated-CV R2, RMSE, MAE + observed vs predicted
  4. Model comparison       : null / GIBSIM-only (linear) / mycobiome-only RF / full RF
  5. Global importance      : impurity (MDI), permutation (out-of-fold), drop-column
  6. SHAP                   : beeswarm, mean|SHAP|, dependence, interaction values
  7. PDP / ICE + 2D PDP     : marginal and joint effects
  8. Stability              : importance ranking across random seeds

Run:  python3 Phytobiome/rf_interpretable_ml.py
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
SEED = 42
TARGET = "FHB_incidence"

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
X = df.drop(columns=TARGET)
y = df[TARGET].values
FEATS = list(X.columns)
FUNGI = [f for f in FEATS if f != "GIBSIM_pred"]
print(f"n = {len(df)}, p = {len(FEATS)}  ->  {FEATS}")

report = []
def log(s=""):
    print(s)
    report.append(s)

# ---------------------------------------------------------------- 1. EDA
log("## 1. EDA")
log(df.describe().T.round(2).to_string())
corr = df.corr(method="spearman")
corr.round(3).to_csv(os.path.join(OUT, "spearman_correlation.csv"))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Spearman rho")
ax.set_title("Spearman correlation")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_correlation.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2. Tuning
log("\n## 2. RF hyper-parameter tuning (5-fold x 5 repeats CV)")
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
grid = {
    "max_features": [1, 2, 3, 4, 7],          # mtry
    "min_samples_leaf": [1, 3, 5, 8],
    "max_depth": [None, 6],
}
gs = GridSearchCV(RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
                  grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
gs.fit(X, y)
best = gs.best_params_
log(f"best params: {best}   CV RMSE = {-gs.best_score_:.2f}")
pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]] \
    .assign(RMSE=lambda d: -d.mean_test_score).sort_values("RMSE") \
    .to_csv(os.path.join(OUT, "tuning_results.csv"), index=False)

rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=SEED,
                           n_jobs=-1, **best).fit(X, y)

# ---------------------------------------------------------------- 3. Performance
log("\n## 3. Performance")
def cv_metrics(model, Xm, n_rep=10):
    r2s, rmses, maes = [], [], []
    preds = np.zeros((n_rep, len(y)))
    for r in range(n_rep):
        p = _oof(model, Xm, r)
        preds[r] = p
        r2s.append(r2_score(y, p)); rmses.append(mean_squared_error(y, p) ** 0.5)
        maes.append(mean_absolute_error(y, p))
    return dict(R2=np.mean(r2s), R2_sd=np.std(r2s), RMSE=np.mean(rmses),
                RMSE_sd=np.std(rmses), MAE=np.mean(maes)), preds.mean(0)

def _oof(model, Xm, seed):
    from sklearn.model_selection import KFold
    p = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xm):
        m = clone(model).fit(Xm.iloc[tr], y[tr])
        p[te] = m.predict(Xm.iloc[te])
    return p

rf_cv, oof_pred = cv_metrics(rf.set_params(oob_score=False, n_estimators=500), X)
rf.set_params(oob_score=True, n_estimators=1000).fit(X, y)
log(f"OOB R2 = {rf.oob_score_:.3f}")
log(f"10x 5-fold CV: R2 = {rf_cv['R2']:.3f} ± {rf_cv['R2_sd']:.3f}, "
    f"RMSE = {rf_cv['RMSE']:.2f} ± {rf_cv['RMSE_sd']:.2f}, MAE = {rf_cv['MAE']:.2f}")
log(f"training R2 (optimistic) = {r2_score(y, rf.predict(X)):.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y, oof_pred, s=25, alpha=.75, color="#2a6f97", edgecolor="white")
ax.plot([0, 100], [0, 100], "--", color="grey")
ax.set_xlabel("Observed FHB incidence (%)"); ax.set_ylabel("CV-predicted (%)")
ax.set_title(f"RF out-of-fold predictions\nR2={rf_cv['R2']:.2f}, RMSE={rf_cv['RMSE']:.1f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_obs_vs_pred.png"), dpi=200); plt.close(fig)

resid = y - oof_pred
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(oof_pred, resid, s=20, alpha=.75); ax.axhline(0, color="grey", ls="--")
ax.set_xlabel("CV-predicted (%)"); ax.set_ylabel("Residual")
ax.set_title("Residuals (RF shrinks extremes toward the mean)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02b_residuals.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 4. Comparison
log("\n## 4. Model comparison (10x 5-fold CV)")
rf_base = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **best)
rf_fungi = clone(rf_base).set_params(max_features=min(best["max_features"], len(FUNGI)))
models = {
    "Null (mean)": (DummyRegressor(), X),
    "Linear: GIBSIM_pred only": (LinearRegression(), X[["GIBSIM_pred"]]),
    "Linear: all features": (LinearRegression(), X),
    "RF: mycobiome only": (rf_fungi, X[FUNGI]),
    "RF: GIBSIM_pred only": (clone(rf_base).set_params(max_features=1), X[["GIBSIM_pred"]]),
    "RF: full (mycobiome + GIBSIM)": (rf_base, X),
}
comp = []
for name, (m, Xm) in models.items():
    met, _ = cv_metrics(m, Xm)
    comp.append(dict(model=name, **{k: round(v, 3) for k, v in met.items()}))
comp = pd.DataFrame(comp)
log(comp.to_string(index=False))
comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

# ---------------------------------------------------------------- 5. Importance
log("\n## 5. Global variable importance")
mdi = pd.Series(rf.feature_importances_, index=FEATS)

# out-of-fold permutation importance (avoids training-set optimism)
perm_rows = []
for r in range(5):
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(X)):
        m = clone(rf_base).fit(X.iloc[tr], y[tr])
        pi = permutation_importance(m, X.iloc[te], y[te], n_repeats=20,
                                    random_state=r * 10 + k, scoring="r2")
        perm_rows.append(pi.importances_mean)
perm = pd.DataFrame(perm_rows, columns=FEATS)

# drop-column importance (handles correlated predictors more honestly)
full_r2 = rf_cv["R2"]
drop = {}
for f in FEATS:
    m = clone(rf_base).set_params(max_features=min(best["max_features"], len(FEATS) - 1))
    met, _ = cv_metrics(m, X.drop(columns=f), n_rep=5)
    drop[f] = full_r2 - met["R2"]
drop = pd.Series(drop)

imp = pd.DataFrame({"MDI": mdi, "Permutation_dR2_mean": perm.mean(),
                    "Permutation_dR2_sd": perm.std(), "DropColumn_dR2": drop})
imp = imp.sort_values("Permutation_dR2_mean", ascending=False)
log(imp.round(4).to_string())
imp.to_csv(os.path.join(OUT, "variable_importance.csv"))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
order = imp.index[::-1]
axes[0].barh(order, imp.loc[order, "MDI"], color="#8d99ae"); axes[0].set_title("Impurity (MDI)")
axes[1].boxplot([perm[f] for f in order], vert=False, tick_labels=order)
axes[1].axvline(0, color="grey", ls="--"); axes[1].set_title("Out-of-fold permutation (ΔR²)")
axes[2].barh(order, imp.loc[order, "DropColumn_dR2"],
             color=["#c1121f" if v > 0 else "#adb5bd" for v in imp.loc[order, "DropColumn_dR2"]])
axes[2].axvline(0, color="grey", ls="--"); axes[2].set_title("Drop-column (ΔCV R²)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_importance.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 6. SHAP
log("\n## 6. SHAP (TreeExplainer)")
expl = shap.TreeExplainer(rf)
sv = expl.shap_values(X)
log(f"base value (mean prediction) = {expl.expected_value if np.isscalar(expl.expected_value) else expl.expected_value[0]:.2f}")
shap_df = pd.DataFrame(sv, columns=FEATS)
shap_df.to_csv(os.path.join(OUT, "shap_values.csv"), index=False)
mean_abs = shap_df.abs().mean().sort_values(ascending=False)
# direction: Spearman between feature value and its SHAP value
direction = pd.Series({f: pd.Series(X[f]).corr(shap_df[f], method="spearman") for f in FEATS})
shap_sum = pd.DataFrame({"mean_abs_SHAP": mean_abs, "rho(feature,SHAP)": direction[mean_abs.index]})
log(shap_sum.round(3).to_string())
shap_sum.to_csv(os.path.join(OUT, "shap_summary.csv"))

plt.figure(); shap.summary_plot(sv, X, show=False)
plt.title("SHAP beeswarm"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_shap_beeswarm.png"), dpi=200, bbox_inches="tight"); plt.close()
plt.figure(); shap.summary_plot(sv, X, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "04b_shap_bar.png"), dpi=200, bbox_inches="tight"); plt.close()

# SHAP interaction values
siv = expl.shap_interaction_values(X)
inter = pd.DataFrame(np.abs(siv).mean(0), index=FEATS, columns=FEATS)
inter.round(3).to_csv(os.path.join(OUT, "shap_interaction_matrix.csv"))
off = inter.where(~np.eye(len(FEATS), dtype=bool)).stack().sort_values(ascending=False)
off = off[[a < b for a, b in off.index]]
log("Top SHAP interactions (mean |phi_ij|, off-diagonal):")
log(off.head(6).round(3).to_string())
main_share = np.diag(inter).sum() / inter.values.sum()
log(f"share of main effects in total |SHAP interaction| mass = {main_share:.1%}")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(inter, cmap="viridis")
ax.set_xticks(range(len(FEATS)), FEATS, rotation=45, ha="right"); ax.set_yticks(range(len(FEATS)), FEATS)
for i in range(len(FEATS)):
    for j in range(len(FEATS)):
        ax.text(j, i, f"{inter.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
fig.colorbar(im, label="mean |SHAP interaction|"); ax.set_title("SHAP interaction values")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "05_shap_interaction.png"), dpi=200); plt.close(fig)

# dependence plots for all features (colour = strongest interacting partner)
ncol = 4; nrow = int(np.ceil(len(FEATS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
for ax, f in zip(axes.ravel(), mean_abs.index):
    partner = inter.loc[f].drop(f).idxmax()
    shap.dependence_plot(f, sv, X, interaction_index=partner, ax=ax, show=False)
    ax.axhline(0, color="grey", lw=.6, ls="--")
for ax in axes.ravel()[len(FEATS):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "06_shap_dependence.png"), dpi=200); plt.close(fig)

# local explanations: lowest, median, highest observed incidence
for tag, idx in {"low": int(np.argmin(y)), "median": int(np.argsort(y)[len(y) // 2]),
                 "high": int(np.argmax(y))}.items():
    plt.figure()
    shap.plots.waterfall(shap.Explanation(sv[idx], expl.expected_value, X.iloc[idx].values, FEATS),
                         show=False)
    plt.title(f"Sample {idx} ({tag}): observed {y[idx]:.1f}%")
    plt.savefig(os.path.join(OUT, f"07_waterfall_{tag}.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 7. PDP / ICE
log("\n## 7. PDP / ICE")
fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
PartialDependenceDisplay.from_estimator(rf, X, list(mean_abs.index), kind="both",
                                        subsample=60, random_state=SEED, ax=ax.ravel()[:len(FEATS)],
                                        ice_lines_kw={"alpha": .15}, pd_line_kw={"color": "red"})
for a in ax.ravel()[len(FEATS):]:
    a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "08_pdp_ice.png"), dpi=200); plt.close(fig)

top_pairs = [off.index[0], off.index[1]]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
PartialDependenceDisplay.from_estimator(rf, X, top_pairs, ax=ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "09_pdp_2d.png"), dpi=200); plt.close(fig)
log(f"2D PDP drawn for {top_pairs}")

# ---------------------------------------------------------------- 8. Stability
log("\n## 8. Importance-rank stability over 30 seeds (mean |SHAP| rank)")
ranks = []
for s in range(30):
    m = clone(rf_base).set_params(random_state=s, n_estimators=300).fit(X, y)
    v = np.abs(shap.TreeExplainer(m).shap_values(X)).mean(0)
    ranks.append(pd.Series(v, index=FEATS).rank(ascending=False))
ranks = pd.DataFrame(ranks)
stab = pd.DataFrame({"mean_rank": ranks.mean(), "min": ranks.min(), "max": ranks.max()}).sort_values("mean_rank")
log(stab.to_string())
stab.to_csv(os.path.join(OUT, "rank_stability.csv"))

with open(os.path.join(OUT, "analysis_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nAll outputs written to {OUT}")
