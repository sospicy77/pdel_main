"""
Causal and asymmetric Shapley values for the FHB random forest
===============================================================
Causal graph assumed (only GIBSIM_pred is available as an environmental variable):

    GIBSIM_pred (weather-driven risk, root)  ->  6 genera (one chain component, mutually
    confounded by unmeasured common causes)  ->  FHB  <-  GIBSIM_pred

Genera model:  genus = a + b * GIBSIM + r,   r ~ MVN(0, Sigma_r)   (fitted by OLS)

Value functions v(S) = expected RF prediction when features in S are known / set:
  marginal     : x_notS drawn from the joint data distribution (standard interventional SHAP)
  conditional  : x_notS ~ p(x_notS | x_S)        (observational; Gaussian residual model,
                                                  GIBSIM | genera by importance resampling)
  causal (do)  : x_notS ~ p(x_notS | do(x_S))    (Heskes et al. 2020; setting a genus does not
                                                  change GIBSIM or the other genera)

Attributions
  SHAP-marginal   : symmetric Shapley, marginal v
  SHAP-conditional: symmetric Shapley, conditional v
  ASV             : asymmetric Shapley (Frye et al. 2020), conditional v, only orderings in
                    which GIBSIM comes before every genus -> GIBSIM receives the credit for the
                    part of each genus that it explains; genera keep only their own part
  Causal SV       : symmetric Shapley, causal (do) v
  Path attribution: GIBSIM's ASV credit routed through genus j
                    = phi_G(ASV) - phi_G(ASV with the GIBSIM -> j edge cut)

Uncertainty: 20 bootstrap refits of the RF and of the genera model (explaining the original
98 samples each time).

Run:  python3 Phytobiome/causal_asymmetric_shapley.py
"""
import itertools
import os
from math import factorial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "causal_shapley")
os.makedirs(OUT, exist_ok=True)
TARGET, G = "FHB_incidence", "GIBSIM_pred"
SEED = 42
N_BOOT = int(os.environ.get("N_BOOT", 20))

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
y = df[TARGET].values
GEN = [c for c in df.columns if c not in (TARGET, G)]
FEATS = [G] + GEN                                   # column order used by the model
X = df[FEATS].values
n, p = X.shape
COL = dict(zip(GEN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
COL[G] = "#52514e"
METHODS = ["SHAP-marginal", "SHAP-conditional", "ASV", "Causal SV"]
MCOL = dict(zip(METHODS, ["#c9c8c2", "#8d8c86", "#2a78d6", "#eb6834"]))

report = []
def log(s=""):
    print(s)
    report.append(str(s))


# ------------------------------------------------------------------ genera model
class GeneraModel:
    def __init__(self, Xfit):
        g = Xfit[:, 0]
        A = np.c_[np.ones(len(g)), g]
        self.coef = np.linalg.lstsq(A, Xfit[:, 1:], rcond=None)[0]          # (2, 6)
        self.res = Xfit[:, 1:] - A @ self.coef
        self.Sig = np.cov(self.res, rowvar=False)
        self.gpool = g
        self.Xpool = Xfit

    def mean(self, g):
        return self.coef[0] + np.outer(g, self.coef[1])                      # (m, 6)


def sample_v(model, gm, Xe, S, kind, M, rng, cut=None):
    """Monte-Carlo E[f] for every row of Xe (n_e, p) and coalition S (indices into FEATS)."""
    S = list(S)
    notS = [j for j in range(p) if j not in S]
    ne = len(Xe)
    Z = np.repeat(Xe, M, axis=0)                         # rows grouped by sample
    if not notS:
        return model.predict(Xe)
    if kind == "marginal":
        rows = rng.integers(0, len(gm.Xpool), ne * M)
        Z[:, notS] = gm.Xpool[rows][:, notS]
        return model.predict(Z).reshape(ne, M).mean(1)

    gS = [j - 1 for j in S if j > 0]                     # genus indices (0..5) in S
    gN = [j - 1 for j in notS if j > 0]
    # --- draw GIBSIM
    if 0 in S:
        gdraw = Z[:, 0].copy()
    elif kind == "causal" or not gS:
        gdraw = gm.gpool[rng.integers(0, len(gm.gpool), ne * M)]
    else:                                                # conditional: p(G | genera_S)
        mu = gm.mean(gm.gpool)[:, gS]                    # (pool, |S|)
        Pinv = np.linalg.inv(gm.Sig[np.ix_(gS, gS)])
        obs = Xe[:, [j + 1 for j in gS]]                 # (ne, |S|)
        d = obs[:, None, :] - mu[None, :, :]             # (ne, pool, |S|)
        logw = -0.5 * np.einsum("ijk,kl,ijl->ij", d, Pinv, d)
        w = np.exp(logw - logw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        cdf = w.cumsum(1)
        u = rng.random((ne, M))
        idx = np.minimum((u[:, :, None] > cdf[:, None, :]).sum(2), len(gm.gpool) - 1)
        gdraw = gm.gpool[idx].ravel()
    Z[:, 0] = gdraw
    if gN:
        mu = gm.mean(gdraw)                              # (ne*M, 6)
        if kind == "causal":                             # do(): residuals ignore intervened genera
            r = gm.res[rng.integers(0, len(gm.res), ne * M)][:, gN]
        else:
            if gS:
                r_S = Z[:, [j + 1 for j in gS]] - mu[:, gS]
                Sss = gm.Sig[np.ix_(gS, gS)]
                Sns = gm.Sig[np.ix_(gN, gS)]
                K = Sns @ np.linalg.inv(Sss)
                cm = r_S @ K.T
                cS = gm.Sig[np.ix_(gN, gN)] - K @ Sns.T
            else:
                cm = np.zeros((ne * M, len(gN)))
                cS = gm.Sig[np.ix_(gN, gN)]
            r = cm + rng.multivariate_normal(np.zeros(len(gN)), (cS + cS.T) / 2, ne * M)
        vals = mu[:, gN] + r
        if cut is not None and cut in gN:                # break GIBSIM -> genus `cut`
            k = gN.index(cut)
            gind = gm.gpool[rng.integers(0, len(gm.gpool), ne * M)]
            vals[:, k] = gm.mean(gind)[:, cut] + r[:, k]
        Z[:, [j + 1 for j in gN]] = vals
    return model.predict(Z).reshape(ne, M).mean(1)


COALS = [frozenset(c) for k in range(p + 1) for c in itertools.combinations(range(p), k)]


def all_values(model, gm, kind, M, seed):
    """v[i, S] for every sample and coalition, batched per coalition for speed."""
    rng = np.random.default_rng(seed)
    V = {S: sample_v(model, gm, X, S, kind, M, rng) for S in COALS}
    V[frozenset()] = np.full(n, V[frozenset()].mean())    # v(empty) is one constant
    return V


def shapley(V, players, given=frozenset()):
    """Shapley values of `players` for the game S -> V[S | given]."""
    k = len(players)
    phi = np.zeros((n, k))
    for a, j in enumerate(players):
        others = [q for q in players if q != j]
        for m in range(k):
            w = factorial(m) * factorial(k - m - 1) / factorial(k)
            for S in itertools.combinations(others, m):
                S = frozenset(S) | given
                phi[:, a] += w * (V[S | {j}] - V[S])
    return phi


def explain(model, gm, M, seed):
    Vm = all_values(model, gm, "marginal", M, seed)
    Vc = all_values(model, gm, "conditional", M, seed + 1)
    Vd = all_values(model, gm, "causal", M, seed + 2)
    gidx = list(range(1, p))
    res = {
        "SHAP-marginal": shapley(Vm, list(range(p))),
        "SHAP-conditional": shapley(Vc, list(range(p))),
        "Causal SV": shapley(Vd, list(range(p))),
    }
    asv = np.zeros((n, p))
    asv[:, 0] = Vc[frozenset({0})] - Vc[frozenset()]
    asv[:, 1:] = shapley(Vc, gidx, given=frozenset({0}))
    res["ASV"] = asv
    res["v0"] = {"SHAP-marginal": Vm[frozenset()][0], "SHAP-conditional": Vc[frozenset()][0],
                 "ASV": Vc[frozenset()][0], "Causal SV": Vd[frozenset()][0]}
    # path attribution: GIBSIM's ASV credit routed through each genus
    rng = np.random.default_rng(seed + 3)
    path = np.zeros((n, len(GEN)))
    for gi in range(len(GEN)):
        v_cut = sample_v(model, gm, X, [0], "conditional", M, rng, cut=gi)
        path[:, gi] = Vc[frozenset({0})] - v_cut
    res["path"] = path
    return res


def summarise(res):
    rows = []
    for mth in METHODS:
        phi = res[mth]
        for j, f in enumerate(FEATS):
            rows.append(dict(method=mth, feature=f, mean_abs=np.abs(phi[:, j]).mean(),
                             rho=pd.Series(X[:, j]).corr(pd.Series(phi[:, j]), method="spearman")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main fit
rf = RandomForestRegressor(n_estimators=500, max_features=3, random_state=SEED, n_jobs=-1).fit(X, y)
gm = GeneraModel(X)
log("Genera model: share of each genus' variance explained by GIBSIM (R2) and slope")
r2g = 1 - gm.res.var(0) / X[:, 1:].var(0)
log(pd.DataFrame({"R2_from_GIBSIM": r2g, "slope_per_10_GIBSIM": gm.coef[1] * 10}, index=GEN).round(3).to_string())

res = explain(rf, gm, M=100, seed=SEED)
f0 = rf.predict(X)
for m_ in METHODS:                                     # efficiency: sum(phi) = f(x) - v(empty)
    gap = np.abs(res[m_].sum(1) - (f0 - res["v0"][m_])).max()
    log(f"{m_:17s}: v(empty) = {res['v0'][m_]:.2f} % (mean RF prediction {f0.mean():.2f}); "
        f"max efficiency gap {gap:.1e}")
summ = summarise(res)
tab = summ.pivot(index="feature", columns="method", values="mean_abs")[METHODS].loc[FEATS]
direc = summ.pivot(index="feature", columns="method", values="rho")[METHODS].loc[FEATS]
log("\nmean |attribution| (FHB %p)")
log(tab.round(2).to_string())
log("\ndirection: Spearman rho(feature value, attribution)")
log(direc.round(2).to_string())

retain = (tab.loc[GEN, "ASV"] / tab.loc[GEN, "SHAP-conditional"]).rename("ASV / SHAP-conditional")
path = pd.Series(np.abs(res["path"]).mean(0), index=GEN, name="GIBSIM credit via genus (mean |.|)")
path_signed = pd.Series([pd.Series(X[:, 0]).corr(pd.Series(res["path"][:, k]), method="spearman")
                         for k in range(len(GEN))], index=GEN, name="rho(GIBSIM, path credit)")
log("\nShare of each genus' conditional-SHAP credit it keeps when GIBSIM is credited first (ASV)")
log(pd.concat([retain, path, path_signed], axis=1).round(3).to_string())
log(f"\nGIBSIM mean |credit|: marginal {tab.loc[G, 'SHAP-marginal']:.2f} -> ASV {tab.loc[G, 'ASV']:.2f} %p "
    f"(sum of path credits {path.sum():.2f})")

# ------------------------------------------------------------------ bootstrap
log(f"\n## Bootstrap ({N_BOOT} refits, M = 40 Monte-Carlo draws)")
rng = np.random.default_rng(SEED)
boot = []
for b in range(N_BOOT):
    ib = rng.integers(0, n, n)
    m = RandomForestRegressor(n_estimators=200, max_features=3, random_state=b, n_jobs=-1).fit(X[ib], y[ib])
    gb = GeneraModel(X[ib])
    rb = explain(m, gb, M=40, seed=100 + b)
    sb = summarise(rb)
    tb = sb.pivot(index="feature", columns="method", values="mean_abs")
    for f in GEN:
        boot.append(dict(b=b, feature=f, ASV=tb.loc[f, "ASV"], SHAPc=tb.loc[f, "SHAP-conditional"],
                         SHAPm=tb.loc[f, "SHAP-marginal"], CSV=tb.loc[f, "Causal SV"],
                         retain=tb.loc[f, "ASV"] / tb.loc[f, "SHAP-conditional"],
                         path=np.abs(rb["path"][:, GEN.index(f)]).mean(),
                         rho_ASV=sb[(sb.method == "ASV") & (sb.feature == f)].rho.iloc[0]))
    print(f"  bootstrap {b + 1}/{N_BOOT} done", flush=True)
boot = pd.DataFrame(boot)
boot.to_csv(os.path.join(OUT, "bootstrap.csv"), index=False)
ci = boot.groupby("feature")[["ASV", "SHAPc", "retain", "path", "rho_ASV"]] \
    .quantile([.025, .5, .975]).unstack().loc[GEN]
log(ci.round(3).to_string())

out_tab = tab.copy()
out_tab.columns = [f"mean_abs_{c}" for c in out_tab.columns]
out_tab = out_tab.join(direc.add_prefix("rho_"))
out_tab.round(4).to_csv(os.path.join(OUT, "attribution_summary.csv"))
pd.concat([retain, path, path_signed], axis=1).round(4).to_csv(os.path.join(OUT, "asv_retention_and_paths.csv"))
for m_ in METHODS + ["path"]:
    cols = FEATS if m_ != "path" else GEN
    pd.DataFrame(res[m_], columns=cols).round(4).to_csv(
        os.path.join(OUT, f"values_{m_.replace(' ', '_')}.csv"), index=False)

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
w = .2
for k, m_ in enumerate(METHODS):
    ax.bar(np.arange(p) + (k - 1.5) * w, tab[m_], w * .92, color=MCOL[m_], label=m_)
ax.set_xticks(range(p), FEATS, rotation=20)
ax.set_ylabel("mean |attribution| (FHB %p)")
ax.set_title("Attribution under four value functions")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
q = boot.groupby("feature")["retain"].quantile([.025, .5, .975]).unstack().loc[GEN]
ax.barh(GEN, retain.loc[GEN] * 100, color=[COL[g] for g in GEN], height=.6)
ax.errorbar(q[.5] * 100, GEN, xerr=[(q[.5] - q[.025]) * 100, (q[.975] - q[.5]) * 100],
            fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
ax.axvline(100, color="#52514e", ls="--", lw=.8)
for yi, v in enumerate(retain.loc[GEN] * 100):
    ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=8)
ax.set_xlabel("credit kept when GIBSIM is credited first\n(ASV / conditional SHAP, %; bootstrap 95% interval)")
ax.set_title("Own (non-weather) share of each genus")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_attribution_methods.png"), dpi=200); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
for ax, f in zip(axes.ravel(), GEN):
    j = FEATS.index(f)
    ax.scatter(X[:, j], res["SHAP-marginal"][:, j], s=14, color="#c9c8c2", label="SHAP-marginal")
    ax.scatter(X[:, j], res["ASV"][:, j], s=20, color=COL[f], edgecolor="white", label="ASV")
    ax.axhline(0, color="#52514e", lw=.6, ls="--")
    ax.set_title(f"{f}: kept {retain[f] * 100:.0f}%, ρ(ASV) = {direc.loc[f, 'ASV']:.2f}", fontsize=9)
    ax.set_xlabel(f"{f} (CLR)")
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("attribution (FHB %p)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Per-sample attribution: standard SHAP (grey) vs asymmetric Shapley with GIBSIM first (colour)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_asv_dependence.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4))
pq = boot.groupby("feature")["path"].quantile([.025, .5, .975]).unstack().loc[GEN]
ax.barh(GEN, path.loc[GEN], color=[COL[g] for g in GEN], height=.6)
ax.errorbar(pq[.5], GEN, xerr=[pq[.5] - pq[.025], pq[.975] - pq[.5]], fmt="none", ecolor="#0b0b0b",
            capsize=3, lw=1)
ax.set_xlabel("mean |GIBSIM credit routed through genus| (FHB %p)")
ax.set_title("Path attribution: GIBSIM → genus → FHB")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_path_attribution.png"), dpi=200); plt.close(fig)

with open(os.path.join(OUT, "causal_shapley_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
