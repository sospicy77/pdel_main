"""
Causal ML check: is each genus a putative suppressive contributor or an ecological indicator?
==============================================================================================
Method (different from the manuscript's Bayesian-GAM g-computation):

1. Double / debiased machine learning (Chernozhukov et al. 2018), partially linear model
       FHB = theta * D + g(W) + e,     D = m(W) + v
   D = CLR abundance of the target genus, W = adjustment set, g and m = random forests,
   5-fold cross-fitting repeated 20 times (median aggregation), heteroskedasticity-robust SE.
   Adjustment sets
     W1 = {GIBSIM_pred}                         confounding by weather-driven risk only
     W2 = {GIBSIM_pred} + other 5 genera        also blocks paths through co-varying taxa
   theta = change in FHB (%p) per +1 CLR unit, "all else in W held fixed".

2. Indicator signature: attenuation of the crude slope after adjustment, and the share of
   the genus' variance explained by W (R2 of m(W)).

3. Sensitivity to unmeasured confounding (Cinelli & Hazlett 2020 omitted-variable bias,
   applied to the DML residual regression):
     RV      = robustness value: partial R2 an unmeasured confounder must have with BOTH
               D and FHB (after W) to drive theta to 0
     RV_a05  = same, to make theta non-significant (alpha = 0.05)
   Benchmark: a confounder "as strong as GIBSIM_pred" (its partial R2 with D and with FHB);
   the script reports the bias-adjusted theta for 1x, 2x and 3x GIBSIM-strength confounding.
   A small RV relative to the GIBSIM benchmark = effect easily explained by unmeasured weather /
   inoculum confounding = consistent with an ecological indicator.

Run:  python3 Phytobiome/causal_dml_sensitivity.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "causal_dml")
os.makedirs(OUT, exist_ok=True)
TARGET, G = "FHB_incidence", "GIBSIM_pred"
TARGETS = ["Alternaria", "Epicoccum", "Hannaella", "Periconia"]
N_REP, K = 20, 5

df = pd.read_excel(os.path.join(HERE, "using_data.xlsx"))
y = df[TARGET].values
n = len(y)
GENERA = [c for c in df.columns if c not in (TARGET, G)]
COL = {"Alternaria": "#2a78d6", "Epicoccum": "#1baf7a", "Hannaella": "#eda100", "Periconia": "#008300"}

report = []
def log(s=""):
    print(s)
    report.append(str(s))


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.6,
                                 random_state=0, n_jobs=-1)


def crossfit_residuals(target, W, seed):
    r = np.zeros(n)
    for tr, te in KFold(K, shuffle=True, random_state=seed).split(W):
        r[te] = target[te] - rf().fit(W[tr], target[tr]).predict(W[te])
    return r


def dml(d, W):
    """Partially linear DML; returns theta, se, and the residuals of the median repetition."""
    out = []
    for s in range(N_REP):
        ry, rd = crossfit_residuals(y, W, s), crossfit_residuals(d, W, s)
        theta = (rd @ ry) / (rd @ rd)
        eps = ry - theta * rd
        se = np.sqrt(np.mean(rd ** 2 * eps ** 2) / np.mean(rd ** 2) ** 2 / n)
        out.append((theta, se, ry, rd))
    th = np.array([o[0] for o in out])
    med = np.median(th)
    se = np.sqrt(np.median([o[1] ** 2 + (o[0] - med) ** 2 for o in out]))   # DML median rule
    i = int(np.argmin(np.abs(th - med)))
    return med, se, out[i][2], out[i][3]


def ovb(ry, rd, theta, se, dof):
    """Cinelli & Hazlett sensitivity statistics from the residual-on-residual regression."""
    t = theta / se
    f = abs(t) / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    tcrit = stats.t.ppf(.975, dof - 1)
    fa = max(f - tcrit / np.sqrt(dof - 1), 0)
    rva = 0.5 * (np.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2)
    return rv, rva


def partial_r2(a, b, ctrl):
    """partial R2 of b with a given ctrl (linear)."""
    X = np.c_[np.ones(n), ctrl] if ctrl is not None else np.ones((n, 1))
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1] ** 2


def adjusted_theta(theta, se, dof, r2dz, r2yz):
    """Bias-adjusted estimate for a confounder with partial R2 r2dz (with D) and r2yz (with Y)."""
    bias = se * np.sqrt(dof) * np.sqrt(r2yz * r2dz / (1 - r2dz))
    se_adj = se * np.sqrt((1 - r2yz) / (1 - r2dz) * dof / (dof - 1))
    return theta - np.sign(theta) * bias, se_adj


rows = []
for g in TARGETS:
    d = df[g].values
    crude = np.polyfit(d, y, 1)[0]
    crude_rho = stats.spearmanr(d, y)[0]
    for wname, wcols in [("W1: GIBSIM", [G]), ("W2: GIBSIM + other genera", [G] + [x for x in GENERA if x != g])]:
        W = df[wcols].values
        theta, se, ry, rd = dml(d, W)
        dof = n - 1 - len(wcols)
        rv, rva = ovb(ry, rd, theta, se, dof)
        r2_d_w = 1 - rd.var() / d.var()                         # how much of D is predictable from W
        # benchmark: GIBSIM's own strength (partial R2 with D and Y given the rest of W)
        rest = df[[c for c in wcols if c != G]].values if len(wcols) > 1 else None
        r2dz = partial_r2(df[G].values, d, rest)
        r2yz = partial_r2(df[G].values, y, np.c_[d, rest] if rest is not None else d[:, None])
        adj = {}
        for kx in (1, 2, 3):
            a, s = adjusted_theta(theta, se, dof, min(kx * r2dz, .99), min(kx * r2yz, .99))
            adj[kx] = (a, s)
        rows.append(dict(genus=g, adjustment=wname, crude_slope=crude, crude_rho=crude_rho,
                         theta=theta, se=se, lo=theta - 1.96 * se, hi=theta + 1.96 * se,
                         p=2 * stats.norm.sf(abs(theta / se)),
                         attenuation_pct=100 * (1 - theta / crude),
                         R2_D_given_W=r2_d_w, RV=rv, RV_alpha05=rva,
                         GIBSIM_partialR2_D=r2dz, GIBSIM_partialR2_Y=r2yz,
                         theta_adj_1xG=adj[1][0], lo_adj_1xG=adj[1][0] - 1.96 * adj[1][1],
                         theta_adj_2xG=adj[2][0], lo_adj_2xG=adj[2][0] - 1.96 * adj[2][1],
                         hi_adj_2xG=adj[2][0] + 1.96 * adj[2][1],
                         theta_adj_3xG=adj[3][0]))

R = pd.DataFrame(rows)
R.round(4).to_csv(os.path.join(OUT, "dml_sensitivity.csv"), index=False)
pd.set_option("display.width", 200)
log("## DML effect (FHB %p per +1 CLR unit) and sensitivity to unmeasured confounding")
log(R[["genus", "adjustment", "crude_slope", "theta", "lo", "hi", "p", "attenuation_pct",
       "R2_D_given_W"]].round(3).to_string(index=False))
log("\nRobustness values (partial R2 an unmeasured confounder needs with BOTH genus and FHB)")
log(R[["genus", "adjustment", "RV", "RV_alpha05", "GIBSIM_partialR2_D", "GIBSIM_partialR2_Y",
       "theta_adj_1xG", "theta_adj_2xG", "lo_adj_2xG", "hi_adj_2xG", "theta_adj_3xG"]]
    .round(3).to_string(index=False))

# ---- figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, wname in zip(axes, R.adjustment.unique()):
    s = R[R.adjustment == wname].reset_index(drop=True)
    for i, r in s.iterrows():
        c = COL[r.genus]
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2.5)
        ax.plot(r.theta, i, "o", color=c, ms=9)
        ax.plot(r.crude_slope, i + .22, "|", color="#52514e", ms=10, mew=2)
        ax.plot([r.lo_adj_2xG, r.hi_adj_2xG], [i - .22, i - .22], color=c, lw=1.2, ls=":")
        ax.plot(r.theta_adj_2xG, i - .22, "D", color="white", mec=c, ms=5)
        ax.text(3.2, i, f"RV={r.RV:.2f}", va="center", fontsize=8,
                color="#52514e")
    ax.axvline(0, color="#52514e", lw=.8)
    ax.set_yticks(range(len(s)), s.genus)
    ax.set_xlabel("Effect on FHB (%p per +1 CLR unit)")
    ax.set_title(f"DML, {wname}\n● DML 95% CI   | crude slope   ◇ bias-adjusted, confounder 2× GIBSIM",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dml_sensitivity.png"), dpi=200); plt.close(fig)

with open(os.path.join(OUT, "dml_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
