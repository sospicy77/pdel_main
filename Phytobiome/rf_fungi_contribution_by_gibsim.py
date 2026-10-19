"""
Relative contribution of the 6 fungal genera to FHB_incidence, by GIBSIM_pred level
=====================================================================================
Question: within each GIBSIM_pred interval, how much of the FHB variation is explained
by (a) each fungus individually, (b) fungus x fungus interactions, (c) GIBSIM_pred,
and how much is left unexplained? And can "contribution x abundance" rebuild FHB?

Part A  RF + out-of-fold SHAP interaction values (non-linear, model-agnostic shares)
        For every sample the OOF prediction is split exactly into
            base + GIBSIM main + sum_k [fungus_k main + fungus_k x GIBSIM] + sum_{k<l} fungus_k x fungus_l
        Within an interval S:
            unexplained              = 1 - R2_S          (R2 of OOF predictions in S)
            share of component c     = R2_S * Cov_S(c, yhat) / Var_S(yhat)
        The shares add up exactly to R2_S, so all parts sum to 100 %.
        "fungus_k" inside an interval = its main effect + its interaction with GIBSIM_pred,
        i.e. the effect of that fungus at that GIBSIM level.

Part B  Interval-specific linear formula   FHB = b0 + sum_k b_k * abundance_k (+ c * GIBSIM)
        Relative contribution = LMG / Shapley decomposition of R2 (relaimpo 'lmg'),
        bootstrap 95 % intervals; b_k are the multipliers for the abundance.

Part C  Can the formula rebuild FHB?  repeated 5-fold CV of
        (i) interval-specific linear formula, (ii) one global linear formula,
        (iii) continuous varying-coefficient model b_k(G) = b_k + g_k * G, vs the RF.

Run:  python3 Phytobiome/rf_fungi_contribution_by_gibsim.py
"""
import itertools
import os
from math import factorial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "contribution_by_gibsim")
os.makedirs(OUT, exist_ok=True)
SEED = 42
TARGET = "FHB_incidence"
G = "GIBSIM_pred"
N_BINS = 3                                           # GIBSIM intervals (quantile-based)
RF_PARAMS = dict(max_features=3, min_samples_leaf=1, max_depth=None)

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
X = df.drop(columns=TARGET)
y = df[TARGET].values
n = len(y)
FEATS = list(X.columns)
FUNGI = [f for f in FEATS if f != G]
gi = FEATS.index(G)
fi = [FEATS.index(f) for f in FUNGI]

# fixed palette: categorical slots 1-6 for the fungi, neutrals for everything else
COL = dict(zip(FUNGI, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
COL.update({"GIBSIM_pred": "#52514e", "Fungus x fungus": "#a3a29c", "Unexplained": "#e4e3df"})

cuts = np.quantile(X[G], np.linspace(0, 1, N_BINS + 1)[1:-1])
bin_id = np.digitize(X[G], cuts)
edges = np.r_[X[G].min(), cuts, X[G].max()]
BIN_NAMES = [f"{['Low', 'Mid', 'High'][k] if N_BINS == 3 else f'Bin {k + 1}'} "
             f"({edges[k]:.0f}–{edges[k + 1]:.0f})" for k in range(N_BINS)]

report = []
def log(s=""):
    print(s)
    report.append(str(s))

log(f"GIBSIM_pred cut points: {np.round(cuts, 1)}  ->  n per interval = {np.bincount(bin_id)}")

# =============================================================== Part A
log("\n## Part A. RF + out-of-fold SHAP interaction decomposition")
N_REP = 5
comp_sum = {k: np.zeros(n) for k in ["GIBSIM_pred", *FUNGI, "Fungus x fungus",
                                     *[f + " (main)" for f in FUNGI], *[f + " x G" for f in FUNGI]]}
yhat = np.zeros(n)
for r in range(N_REP):
    for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
        m = RandomForestRegressor(n_estimators=300, random_state=r, n_jobs=-1, **RF_PARAMS) \
            .fit(X.iloc[tr], y[tr])
        ex = shap.TreeExplainer(m)
        phi = ex.shap_interaction_values(X.iloc[te])                 # (nt, p, p)
        base = float(np.ravel(ex.expected_value)[0])
        yhat[te] += base + phi.sum((1, 2))
        comp_sum["GIBSIM_pred"][te] += phi[:, gi, gi]
        for f, j in zip(FUNGI, fi):
            main, xg = phi[:, j, j], phi[:, j, gi] + phi[:, gi, j]
            comp_sum[f + " (main)"][te] += main
            comp_sum[f + " x G"][te] += xg
            comp_sum[f][te] += main + xg
        ff = sum(phi[:, a, b] for a in fi for b in fi if a != b)
        comp_sum["Fungus x fungus"][te] += ff
comp = {k: v / N_REP for k, v in comp_sum.items()}
yhat /= N_REP


def shares(idx, keys):
    """Variance shares (fractions of Var(y) within idx) for components `keys` + unexplained."""
    yy, yh = y[idx], yhat[idx]
    r2 = 1 - ((yy - yh) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    r2c = max(r2, 0.0)
    vh = yh.var()
    out = {k: r2c * np.cov(comp[k][idx], yh, bias=True)[0, 1] / vh for k in keys}
    out["Unexplained"] = 1 - r2c
    return out, r2


KEYS_BIN = ["GIBSIM_pred", *FUNGI, "Fungus x fungus"]
rows, ci_rows = [], []
groups = [("All samples", np.arange(n))] + [(BIN_NAMES[k], np.where(bin_id == k)[0])
                                            for k in range(N_BINS)]
rng = np.random.default_rng(SEED)
for name, idx in groups:
    s, r2 = shares(idx, KEYS_BIN)
    s["Fungi individual (sum)"] = sum(s[f] for f in FUNGI)
    s.update(dict(group=name, n=len(idx), R2_oof=r2, Var_y=y[idx].var(),
                  mean_FHB=y[idx].mean()))
    rows.append(s)
    boots = []                                                       # sampling uncertainty
    for _ in range(500):
        b = rng.choice(idx, len(idx), replace=True)
        sb, _ = shares(b, KEYS_BIN)
        sb["Fungi individual (sum)"] = sum(sb[f] for f in FUNGI)
        boots.append(sb)
    boots = pd.DataFrame(boots)
    ci_rows.append(pd.concat({"lo": boots.quantile(.025), "hi": boots.quantile(.975)}, axis=1)
                   .assign(group=name))
A = pd.DataFrame(rows).set_index("group")
ci = pd.concat(ci_rows)
order_cols = ["n", "mean_FHB", "Var_y", "R2_oof", "GIBSIM_pred", *FUNGI,
              "Fungi individual (sum)", "Fungus x fungus", "Unexplained"]
A = A[order_cols]
pct = A.copy()
pct.iloc[:, 4:] *= 100
log("Shares in % of FHB variance WITHIN each group (sum of GIBSIM + 6 fungi + fungus x fungus + "
    "unexplained = 100)")
log(pct.round(1).T.to_string())
pct.round(2).to_csv(os.path.join(OUT, "A_variance_shares_by_gibsim.csv"))
ci.round(4).to_csv(os.path.join(OUT, "A_variance_shares_bootstrap_CI.csv"))

# whole-data split of each fungus into main vs x GIBSIM part
s_all, _ = shares(np.arange(n), [*[f + " (main)" for f in FUNGI], *[f + " x G" for f in FUNGI]])
split = pd.DataFrame({"main": [s_all[f + " (main)"] for f in FUNGI],
                      "x GIBSIM": [s_all[f + " x G"] for f in FUNGI]}, index=FUNGI) * 100
log("\nAll samples: each fungus split into main effect vs interaction with GIBSIM_pred (% of Var)")
log(split.round(1).to_string())

# relative contribution among fungi only (normalised to 100 % of the fungal individual part)
rel = A[FUNGI].div(A[FUNGI].clip(lower=0).sum(1), axis=0) * 100
log("\nRelative contribution among the 6 fungi (% of the fungal individual part, negatives kept)")
log(rel.round(1).to_string())
rel.round(2).to_csv(os.path.join(OUT, "A_relative_contribution_among_fungi.csv"))

# component signs: direction of each fungus' effect inside every interval
dirn = pd.DataFrame({name: [np.corrcoef(X[f].values[idx], comp[f][idx])[0, 1] for f in FUNGI]
                     for name, idx in groups}, index=FUNGI)
log("\nDirection: corr(abundance, its SHAP contribution) inside each group (+ raises FHB)")
log(dirn.round(2).to_string())
dirn.round(3).to_csv(os.path.join(OUT, "A_direction_by_gibsim.csv"))

# ---- figure A: stacked composition + per-fungus shares
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
stack = ["GIBSIM_pred", *FUNGI, "Fungus x fungus", "Unexplained"]
ylab = list(A.index)[::-1]
for yi, g in enumerate(ylab):
    left_pos, left_neg = 0.0, 0.0
    for k in stack:
        v = pct.loc[g, k]
        if v >= 0:
            ax.barh(yi, v, left=left_pos, color=COL[k], edgecolor="white", linewidth=2, height=.6)
            if v >= 4:
                ax.text(left_pos + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="#0b0b0b" if k in ("Unexplained", "Fungus x fungus") or k in FUNGI[2:5]
                        else "white")
            left_pos += v
        else:
            ax.barh(yi, v, left=left_neg, color=COL[k], edgecolor="white", linewidth=2,
                    height=.6, hatch="///")
            left_neg += v
ax.set_yticks(range(len(ylab)), [f"{g}\nn={int(A.loc[g, 'n'])}, R²={A.loc[g, 'R2_oof']:.2f}"
                                  for g in ylab])
ax.axvline(0, color="#52514e", lw=.8)
ax.set_xlabel("% of FHB variance within group  (hatched = negative share)")
ax.set_title("Variance partition by GIBSIM_pred interval")
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in stack]
ax.legend(handles, stack, ncol=5, fontsize=8, frameon=False, loc="upper center",
          bbox_to_anchor=(.5, -.16))
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
w = .8 / N_BINS
for b in range(N_BINS):
    g = BIN_NAMES[b]
    lo = ci[(ci.group == g)].loc[FUNGI, "lo"] * 100
    hi = ci[(ci.group == g)].loc[FUNGI, "hi"] * 100
    v = pct.loc[g, FUNGI]
    xs = np.arange(len(FUNGI)) + (b - (N_BINS - 1) / 2) * w
    ax.bar(xs, v, width=w * .9, color=[COL[f] for f in FUNGI], alpha=[.45, .7, 1][b] if N_BINS == 3 else 1,
           edgecolor="white", linewidth=1)
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#52514e", lw=.8, capsize=2)
ax.set_xticks(range(len(FUNGI)), FUNGI, rotation=20)
ax.axhline(0, color="#52514e", lw=.8)
ax.set_ylabel("% of within-interval FHB variance")
ax.set_title("Individual fungal shares: light → dark = Low → Mid → High GIBSIM\n"
             "(bootstrap 95% interval)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "A_variance_partition.png"), dpi=200,
                                bbox_inches="tight"); plt.close(fig)

# =============================================================== Part B
log("\n## Part B. Interval-specific linear formula + LMG (Shapley R2) relative importance")


def r2_ols(Xm, yy):
    Xd = np.c_[np.ones(len(yy)), Xm]
    beta, *_ = np.linalg.lstsq(Xd, yy, rcond=None)
    res = yy - Xd @ beta
    return 1 - res @ res / ((yy - yy.mean()) ** 2).sum(), beta


def lmg(Xm, yy):
    p = Xm.shape[1]
    r2 = {(): 0.0}
    for k in range(1, p + 1):
        for S in itertools.combinations(range(p), k):
            r2[S] = r2_ols(Xm[:, S], yy)[0]
    out = np.zeros(p)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            wgt = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in itertools.combinations(others, k):
                out[j] += wgt * (r2[tuple(sorted(S + (j,)))] - r2[S])
    return out


B_rows, coef_rows = [], []
for name, idx in groups:
    for with_g in (False, True):
        cols = FUNGI + ([G] if with_g else [])
        Xm, yy = X[cols].values[idx], y[idx]
        r2, beta = r2_ols(Xm, yy)
        imp = lmg(Xm, yy)
        bt = []
        for _ in range(200):
            b = rng.integers(0, len(idx), len(idx))
            bt.append(np.r_[lmg(Xm[b], yy[b]), r2_ols(Xm[b], yy[b])[1]])
        bt = np.array(bt)
        q = np.quantile(bt, [.025, .975], axis=0)
        model = "fungi + GIBSIM" if with_g else "fungi only"
        for j, c in enumerate(cols):
            B_rows.append(dict(group=name, model=model, variable=c, LMG_pct=imp[j] * 100,
                               LMG_lo=q[0, j] * 100, LMG_hi=q[1, j] * 100))
            coef_rows.append(dict(group=name, model=model, variable=c, coef=beta[j + 1],
                                  lo=q[0, len(cols) + j + 1], hi=q[1, len(cols) + j + 1]))
        coef_rows.append(dict(group=name, model=model, variable="(intercept)", coef=beta[0],
                              lo=q[0, len(cols)], hi=q[1, len(cols)]))
        B_rows.append(dict(group=name, model=model, variable="R2 (in-sample)", LMG_pct=r2 * 100))
B = pd.DataFrame(B_rows)
C = pd.DataFrame(coef_rows)
B.round(2).to_csv(os.path.join(OUT, "B_LMG_relative_importance.csv"), index=False)
C.round(3).to_csv(os.path.join(OUT, "B_linear_coefficients.csv"), index=False)
for model in ["fungi only", "fungi + GIBSIM"]:
    log(f"\nLMG relative importance (% of FHB variance within group), model: {model}")
    log(B[B.model == model].pivot(index="variable", columns="group", values="LMG_pct")
        [[g for g, _ in groups]].round(1).to_string())
log("\nLinear multipliers b_k (FHB %p per +1 unit abundance), model: fungi only")
log(C[C.model == "fungi only"].pivot(index="variable", columns="group", values="coef")
    [[g for g, _ in groups]].round(2).to_string())

# ---- figure B: coefficients by interval
fig, axes = plt.subplots(1, len(FUNGI), figsize=(17, 3.6), sharey=False)
cf = C[C.model == "fungi only"]
for ax, f in zip(axes, FUNGI):
    d = cf[(cf.variable == f) & (cf.group != "All samples")].set_index("group").loc[BIN_NAMES]
    xs = np.arange(N_BINS)
    ax.errorbar(xs, d.coef, yerr=[d.coef - d.lo, d.hi - d.coef], fmt="o-", color=COL[f],
                ms=7, lw=2, capsize=3)
    ax.axhline(0, color="#52514e", lw=.8, ls="--")
    ax.set_xticks(xs, [b.split(" ")[0] for b in BIN_NAMES])
    ax.set_title(f, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("b_k: FHB %p per unit abundance")
fig.suptitle("Interval-specific multipliers (fungi-only linear formula, bootstrap 95% interval) "
             "across GIBSIM_pred intervals", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B_linear_multipliers.png"), dpi=200,
                                bbox_inches="tight"); plt.close(fig)

# =============================================================== Part C
log("\n## Part C. Can 'multiplier x abundance' rebuild FHB?  (10 x 5-fold CV, predictions clipped 0-100)")


def ols_fit_predict(Xtr, ytr, Xte):
    beta = np.linalg.lstsq(np.c_[np.ones(len(ytr)), Xtr], ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ beta


def vc_design(Xd):                                      # varying-coefficient: x_k, G, x_k * G
    Gc = Xd[G].values[:, None]
    return np.c_[Xd[FUNGI].values, Gc, Xd[FUNGI].values * Gc]


def cv_r2(predict_fn, n_rep=10):
    out = []
    for r in range(n_rep):
        p = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            p[te] = np.clip(predict_fn(tr, te), 0, 100)
        out.append(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.mean(out), np.std(out), p


def binwise(tr, te, with_g):
    cols = FUNGI + ([G] if with_g else [])
    p = np.zeros(len(te))
    for b in range(N_BINS):
        trb, teb = tr[bin_id[tr] == b], np.where(bin_id[te] == b)[0]
        if len(teb):
            p[teb] = ols_fit_predict(X[cols].values[trb], y[trb], X[cols].values[te[teb]])
    return p


cands = {
    "Global linear: fungi only": lambda tr, te: ols_fit_predict(X[FUNGI].values[tr], y[tr],
                                                                X[FUNGI].values[te]),
    "Global linear: fungi + GIBSIM": lambda tr, te: ols_fit_predict(X.values[tr], y[tr], X.values[te]),
    f"Interval-wise linear ({N_BINS} bins): fungi only": lambda tr, te: binwise(tr, te, False),
    f"Interval-wise linear ({N_BINS} bins): fungi + GIBSIM": lambda tr, te: binwise(tr, te, True),
    "Varying-coef linear: b_k(G)=b_k+g_k*G": lambda tr, te: ols_fit_predict(
        vc_design(X.iloc[tr]), y[tr], vc_design(X.iloc[te])),
    "Random forest (all 7)": lambda tr, te: RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1, **RF_PARAMS)
        .fit(X.iloc[tr], y[tr]).predict(X.iloc[te]),
}
C_rows, preds = [], {}
for name, fn in cands.items():
    m, s, p = cv_r2(fn, n_rep=10 if "forest" not in name else 3)
    C_rows.append(dict(model=name, CV_R2=m, CV_R2_sd=s,
                       CV_RMSE=np.sqrt(((y - p) ** 2).mean())))
    preds[name] = p
Ct = pd.DataFrame(C_rows)
log(Ct.round(3).to_string(index=False))
Ct.round(4).to_csv(os.path.join(OUT, "C_formula_vs_RF_cv.csv"), index=False)

# final formula table for practical use (fit on all data, fungi only, per interval)
form = C[(C.model == "fungi only") & (C.group != "All samples")] \
    .pivot(index="group", columns="variable", values="coef").loc[BIN_NAMES][["(intercept)", *FUNGI]]
form.round(3).to_csv(os.path.join(OUT, "C_formula_coefficients_per_interval.csv"))
log("\nFHB_hat = intercept + sum_k b_k * abundance_k   (per GIBSIM_pred interval)")
log(form.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, name in zip(axes, [f"Interval-wise linear ({N_BINS} bins): fungi only", "Random forest (all 7)"]):
    for b in range(N_BINS):
        m = bin_id == b
        ax.scatter(y[m], preds[name][m], s=28, color=["#9ec5f0", "#2a78d6", "#0d3b75"][b],
                   edgecolor="white", label=BIN_NAMES[b])
    ax.plot([0, 100], [0, 100], ls="--", color="#52514e", lw=.8)
    r2v = Ct.set_index("model").loc[name, "CV_R2"]
    ax.set_title(f"{name}\nCV R² = {r2v:.2f}", fontsize=10)
    ax.set_xlabel("Observed FHB incidence (%)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("CV-predicted (%)")
axes[0].legend(title="GIBSIM_pred", fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "C_formula_vs_RF.png"), dpi=200); plt.close(fig)

with open(os.path.join(OUT, "contribution_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
