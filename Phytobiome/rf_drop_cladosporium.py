"""
Does Epicoccum become important once Cladosporium is removed?
=============================================================
Same RF pipeline applied to paired feature sets (with vs without Cladosporium):
  Set 1: all 7 inputs                 vs  6 inputs (no Cladosporium)
  Set 2: 5 fungi (no GIBSIM/Alternaria) vs 4 fungi (no Cladosporium)

For each model: tuned RF, 10x 5-fold CV R2, OOF permutation importance, drop-column
importance (paired CV difference), mean |SHAP| + rank, rank stability over 20 seeds,
ALE range of Epicoccum, and the paired CV gain from Epicoccum.

Run:  python3 Phytobiome/rf_drop_cladosporium.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "drop_cladosporium")
os.makedirs(OUT, exist_ok=True)
SEED = 42
TARGET = "FHB_incidence"
E, C = "Epicoccum", "Cladosporium"
N_REP = 10

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
y = df[TARGET].values
n = len(y)
ALL = [c for c in df.columns if c != TARGET]
FIVE = [c for c in ALL if c not in ("GIBSIM_pred", "Alternaria")]
SETS = {
    "7 vars (with Cladosporium)": ALL,
    "6 vars (no Cladosporium)": [c for c in ALL if c != C],
    "5 fungi (with Cladosporium)": FIVE,
    "4 fungi (no Cladosporium)": [c for c in FIVE if c != C],
}

report = []
def log(s=""):
    print(s)
    report.append(str(s))


def oof(model, Xm, seed):
    p = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xm):
        p[te] = clone(model).fit(Xm.iloc[tr], y[tr]).predict(Xm.iloc[te])
    return p


def cv_r2(model, Xm):
    return np.array([r2_score(y, oof(model, Xm, r)) for r in range(N_REP)])


def ale_range(model, Xd, f, bins=10):
    z = np.unique(np.quantile(Xd[f], np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(Xd[f], z[1:-1]), 0, len(z) - 2)
    eff = []
    for k in range(len(z) - 1):
        m = idx == k
        lo, hi = Xd[m].copy(), Xd[m].copy()
        lo[f], hi[f] = z[k], z[k + 1]
        eff.append((model.predict(hi) - model.predict(lo)).mean() if m.any() else 0)
    acc = np.r_[0, np.cumsum(eff)]
    return z, acc - acc.mean(), acc.max() - acc.min(), acc[-1] - acc[0]


summary, imp_tables, ale_curves = [], {}, {}
for name, feats in SETS.items():
    log(f"\n## {name}: {feats}")
    X = df[feats]
    p = len(feats)
    gs = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                      {"max_features": sorted({1, 2, 3, p}), "min_samples_leaf": [1, 3, 5]},
                      cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED),
                      scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)
    best = gs.best_params_
    base = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **best)
    r2 = cv_r2(base, X)
    log(f"best params {best};  CV R2 = {r2.mean():.3f} ± {r2.std():.3f}")

    perm = []
    for r in range(3):
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(X)):
            m = clone(base).fit(X.iloc[tr], y[tr])
            perm.append(permutation_importance(m, X.iloc[te], y[te], n_repeats=15,
                                               random_state=r * 10 + k, scoring="r2").importances_mean)
    perm = pd.DataFrame(perm, columns=feats)

    drop, drop_pos = {}, {}
    for f in feats:
        m = clone(base).set_params(max_features=min(best["max_features"], p - 1))
        d = r2 - cv_r2(m, X.drop(columns=f))                      # paired by CV split seed
        drop[f], drop_pos[f] = d.mean(), int((d > 0).sum())

    rf = RandomForestRegressor(n_estimators=1000, random_state=SEED, n_jobs=-1, **best).fit(X, y)
    S = pd.DataFrame(shap.TreeExplainer(rf).shap_values(X), columns=feats)
    ranks = []
    for s in range(20):
        ms = clone(base).set_params(random_state=s).fit(X, y)
        v = np.abs(shap.TreeExplainer(ms).shap_values(X)).mean(0)
        ranks.append(pd.Series(v, index=feats).rank(ascending=False))
    ranks = pd.DataFrame(ranks)

    imp = pd.DataFrame({
        "Perm_dR2": perm.mean(), "Perm_sd": perm.std(),
        "Drop_dR2": pd.Series(drop), "Drop_positive_of_10": pd.Series(drop_pos),
        "mean_abs_SHAP": S.abs().mean(),
        "rho(value,SHAP)": [X[f].corr(S[f], method="spearman") for f in feats],
        "SHAP_rank_mean_20seeds": ranks.mean(),
    }).sort_values("mean_abs_SHAP", ascending=False)
    imp["rank_SHAP"] = np.arange(1, p + 1)
    imp["rank_Perm"] = imp.Perm_dR2.rank(ascending=False).astype(int)
    log(imp.round(3).to_string())
    imp.round(4).to_csv(os.path.join(OUT, f"importance_{name.split(' (')[0].replace(' ', '_')}"
                                          f"_{'with' if C in feats else 'no'}_clado.csv"))
    imp_tables[name] = imp

    z, ale, span, net = ale_range(rf, X, E)
    ale_curves[name] = (z, ale)
    log(f"Epicoccum ALE: span {span:.1f} %p, net low->high {net:+.1f} %p")
    summary.append(dict(model=name, CV_R2=r2.mean(), CV_R2_sd=r2.std(),
                        Epi_rank_SHAP=int(imp.loc[E, "rank_SHAP"]), Epi_rank_Perm=int(imp.loc[E, "rank_Perm"]),
                        n_vars=p, Epi_mean_abs_SHAP=imp.loc[E, "mean_abs_SHAP"],
                        Epi_Perm_dR2=imp.loc[E, "Perm_dR2"], Epi_Drop_dR2=imp.loc[E, "Drop_dR2"],
                        Epi_Drop_positive_of_10=imp.loc[E, "Drop_positive_of_10"],
                        Epi_SHAP_direction=imp.loc[E, "rho(value,SHAP)"],
                        Epi_rank_20seeds=imp.loc[E, "SHAP_rank_mean_20seeds"],
                        Epi_ALE_span=span, Epi_ALE_net=net))

summ = pd.DataFrame(summary).set_index("model")
log("\n## Summary for Epicoccum")
log(summ.round(3).T.to_string())
summ.round(4).to_csv(os.path.join(OUT, "epicoccum_summary.csv"))

# ---- figure: Epicoccum importance with vs without Cladosporium
names = list(SETS)
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
cols = ["#9ec5f0", "#2a78d6", "#f6c3a8", "#eb6834"]
for ax, metric, title in zip(axes[:3], ["Epi_mean_abs_SHAP", "Epi_Perm_dR2", "Epi_Drop_dR2"],
                             ["Epicoccum mean |SHAP| (%p)", "Epicoccum OOF permutation ΔR²",
                              "Epicoccum drop-column ΔR²"]):
    v = summ.loc[names, metric]
    ax.bar(range(4), v, color=cols, edgecolor="white", width=.7)
    for i, (val, nm) in enumerate(zip(v, names)):
        ax.text(i, val, f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8)
        ax.text(i, 0, f"rank {summ.loc[nm, 'Epi_rank_SHAP']}/{summ.loc[nm, 'n_vars']}",
                ha="center", va="top" if val >= 0 else "bottom", fontsize=7, color="#52514e")
    ax.axhline(0, color="#52514e", lw=.8)
    ax.set_xticks(range(4), [nm.replace(" (", "\n(") for nm in names], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
ax = axes[3]
for nm, c in zip(names, cols):
    z, ale = ale_curves[nm]
    ax.plot(z, ale, "-o", ms=3, lw=2, color=c, label=nm)
ax.axhline(0, color="#52514e", lw=.8, ls="--")
ax.set_xlabel(E); ax.set_ylabel("ALE (%p)"); ax.set_title("Epicoccum ALE", fontsize=10)
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "epicoccum_with_vs_without_clado.png"), dpi=200)
plt.close(fig)

with open(os.path.join(OUT, "drop_cladosporium_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
