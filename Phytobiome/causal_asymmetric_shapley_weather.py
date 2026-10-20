"""
Causal / asymmetric Shapley values with measured flowering-stage weather (using_df_2.xlsx)
===========================================================================================
Causal graph assumed:

    Environment block E = {flowering temperature, flowering RH, GIBSIM, post-sampling rain}
        (one root component; members mutually dependent through site/season)
    E  ->  6 genera (one chain component, mutually confounded)  ->  FHB  <-  E

Post-sampling rain cannot change the flowering mycobiome; it sits in E because it shares
common causes (site, season) with it and acts directly on FHB.

Models
  FHB model     : random forest on all inputs (tuned by repeated CV)
  genera | E    : genus = a + B E + r,  r ~ MVN(0, Sigma_r)   (OLS)
  E | (...)     : empirical rows of E, importance-weighted by a Gaussian kernel on known
                  E members (bandwidth 0.5 SD) and by the likelihood of known genera

Value functions v(S)
  marginal    : x_notS from the joint data distribution (standard interventional SHAP)
  conditional : x_notS ~ p(x_notS | x_S)                 (observational)
  causal (do) : x_notS ~ p(x_notS | do(x_S))             (Heskes et al. 2020)

Attributions (all exact over coalitions)
  SHAP-marginal, SHAP-conditional, Causal SV : symmetric Shapley over all 2^p coalitions
  ASV env-first     : environment block credited before any genus (Frye et al. 2020)
  ASV genera-first  : genera credited before the environment (the opposite claim)
  kept share        = ASV env-first / ASV genera-first   (credit a genus keeps when weather
                      is allowed to explain it first)
  path attribution  : environment credit routed through genus j
                      = v(E) - v(E with the E -> genus j edge cut)

Bootstrap: 20 refits of the RF and the genera model for the two ASVs and the paths.

Environment sets (ENV_SET environment variable):
  full (default) : temperature, RH, GIBSIM, post-sampling rain  -> output/causal_shapley_weather
  TRH            : flowering temperature and RH only            -> output/causal_shapley_weather_TRH

Run:  python3 Phytobiome/causal_asymmetric_shapley_weather.py
      ENV_SET=TRH python3 Phytobiome/causal_asymmetric_shapley_weather.py
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_SET = os.environ.get("ENV_SET", "full")
ENV_SETS = {
    "full": ["temp_flower_sampling_10days", "rhum_flower_sampling_10days", "GIBSIM", "prcp_after_sampling"],
    "TRH": ["temp_flower_sampling_10days", "rhum_flower_sampling_10days"],
}
OUT = os.path.join(HERE, "output", "causal_shapley_weather" + ("" if ENV_SET == "full" else f"_{ENV_SET}"))
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_BOOT = int(os.environ.get("N_BOOT", 20))
M_MAIN = int(os.environ.get("M_MAIN", 40))
TARGET = "FHB_incidence"

df = pd.read_excel(os.path.join(HERE, "using_df_2.xlsx"))
ENV = ENV_SETS[ENV_SET]
ENV_SHORT = {"temp_flower_sampling_10days": "Temp (flowering)", "rhum_flower_sampling_10days": "RH (flowering)",
             "GIBSIM": "GIBSIM", "prcp_after_sampling": "Rain (after sampling)"}
GEN = ["Alternaria", "Cladosporium", "Epicoccum", "Hannaella", "Papiliotrema", "Periconia"]
FEATS = ENV + GEN
X = df[FEATS].values
y = df[TARGET].values
n, p = X.shape
NE = len(ENV)
EI, GI = list(range(NE)), list(range(NE, p))
LABEL = [ENV_SHORT[f] for f in ENV] + GEN
COL = dict(zip(GEN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
METHODS = ["SHAP-marginal", "SHAP-conditional", "Causal SV", "ASV genera-first", "ASV env-first"]
MCOL = dict(zip(METHODS, ["#d9d8d2", "#a3a29c", "#eb6834", "#9ec5f0", "#2a78d6"]))

report = []
def log(s=""):
    print(s)
    report.append(str(s))


# ------------------------------------------------------------------ models
class GeneraModel:
    def __init__(self, Xfit):
        E = Xfit[:, EI]
        A = np.c_[np.ones(len(E)), E]
        self.coef = np.linalg.lstsq(A, Xfit[:, GI], rcond=None)[0]           # (1+NE, 6)
        self.res = Xfit[:, GI] - A @ self.coef
        self.Sig = np.cov(self.res, rowvar=False)
        self.Epool = E
        self.Esd = E.std(0)
        self.Xpool = Xfit

    def mean(self, E):
        return self.coef[0] + E @ self.coef[1:]


def draw_rows(logw, M, rng):
    """For each row of logw (ne, pool) draw M pool indices."""
    w = np.exp(logw - logw.max(1, keepdims=True))
    cdf = (w / w.sum(1, keepdims=True)).cumsum(1)
    u = rng.random((len(logw), M))
    return np.minimum((u[:, :, None] > cdf[:, None, :]).sum(2), logw.shape[1] - 1)


def sample_v(model, gm, Xe, S, kind, M, rng, cut=None):
    S = set(S)
    notS = [j for j in range(p) if j not in S]
    ne = len(Xe)
    if not notS:
        return model.predict(Xe)
    Z = np.repeat(Xe, M, axis=0)
    if kind == "marginal":
        Z[:, notS] = gm.Xpool[rng.integers(0, len(gm.Xpool), ne * M)][:, notS]
        return model.predict(Z).reshape(ne, M).mean(1)

    eS = [j for j in EI if j in S]
    eN = [j for j in EI if j not in S]
    gS = [j - NE for j in GI if j in S]
    gN = [j - NE for j in GI if j not in S]
    npool = len(gm.Epool)
    # ---------------- environment block
    if eN:
        if kind == "causal" or (not eS and not gS):
            rows = rng.integers(0, npool, (ne, M))
        else:
            logw = np.zeros((ne, npool))
            if eS:
                dz = (gm.Epool[None, :, eS] - Xe[:, None, eS]) / (0.5 * gm.Esd[eS])
                logw += -0.5 * (dz ** 2).sum(2)
            if gS:
                mu = gm.mean(gm.Epool)[:, gS]                                  # (pool, |gS|)
                Pinv = np.linalg.inv(gm.Sig[np.ix_(gS, gS)])
                d = Xe[:, None, [NE + k for k in gS]] - mu[None]
                logw += -0.5 * np.einsum("ijk,kl,ijl->ij", d, Pinv, d)
            rows = draw_rows(logw, M, rng)
        Z[:, eN] = gm.Epool[rows.ravel()][:, eN]
    # ---------------- genera
    if gN:
        E = Z[:, EI]
        mu = gm.mean(E)
        if kind == "causal":
            r = gm.res[rng.integers(0, len(gm.res), ne * M)][:, gN]
        else:
            if gS:
                r_S = Z[:, [NE + k for k in gS]] - mu[:, gS]
                Sss, Sns = gm.Sig[np.ix_(gS, gS)], gm.Sig[np.ix_(gN, gS)]
                K = Sns @ np.linalg.inv(Sss)
                cm, cS = r_S @ K.T, gm.Sig[np.ix_(gN, gN)] - K @ Sns.T
            else:
                cm, cS = np.zeros((ne * M, len(gN))), gm.Sig[np.ix_(gN, gN)]
            r = cm + rng.multivariate_normal(np.zeros(len(gN)), (cS + cS.T) / 2, ne * M)
        vals = mu[:, gN] + r
        if cut is not None and cut in gN:                    # break E -> genus `cut`
            k = gN.index(cut)
            Eind = gm.Epool[rng.integers(0, npool, ne * M)]
            vals[:, k] = gm.mean(Eind)[:, cut] + r[:, k]
        Z[:, [NE + k for k in gN]] = vals
    return model.predict(Z).reshape(ne, M).mean(1)


class Game:
    """Caches v(S) for one value function."""
    def __init__(self, model, gm, kind, M, seed):
        self.model, self.gm, self.kind, self.M = model, gm, kind, M
        self.rng = np.random.default_rng(seed)
        self.V = {}

    def __call__(self, S):
        S = frozenset(S)
        if S not in self.V:
            self.V[S] = sample_v(self.model, self.gm, X, S, self.kind, self.M, self.rng)
            if not S:
                self.V[S] = np.full(n, self.V[S].mean())
        return self.V[S]


def shapley(v, players, given=frozenset()):
    k = len(players)
    phi = np.zeros((n, k))
    for a, j in enumerate(players):
        others = [q for q in players if q != j]
        for m in range(k):
            w = factorial(m) * factorial(k - m - 1) / factorial(k)
            for S in itertools.combinations(others, m):
                S = frozenset(S) | given
                phi[:, a] += w * (v(S | {j}) - v(S))
    return phi


def asv(v, first, second):
    phi = np.zeros((n, p))
    phi[:, first] = shapley(v, first)
    phi[:, second] = shapley(v, second, given=frozenset(first))
    return phi


def paths(model, gm, vcond, M, seed):
    rng = np.random.default_rng(seed)
    base = vcond(EI)
    out = np.zeros((n, len(GEN)))
    for k in range(len(GEN)):
        out[:, k] = base - sample_v(model, gm, X, EI, "conditional", M, rng, cut=k)
    return out


# ------------------------------------------------------------------ fit
log(f"## Random forest on {p} inputs (environment set: {ENV_SET} = {ENV})")
gs = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                  {"max_features": sorted({2, 3, 4, 6, p}), "min_samples_leaf": [1, 3, 5]},
                  cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED),
                  scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)
best = gs.best_params_


def cv_r2(cols, reps=5):
    out = []
    for r in range(reps):
        pr = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1,
                                      max_features=min(best["max_features"], len(cols)),
                                      min_samples_leaf=best["min_samples_leaf"]).fit(X[tr][:, cols], y[tr])
            pr[te] = m.predict(X[te][:, cols])
        out.append(r2_score(y, pr))
    return np.mean(out), np.std(out)


for name, cols in [(f"environment only ({NE})", EI), ("genera only (6)", GI), (f"all {p}", list(range(p)))]:
    m_, s_ = cv_r2(cols)
    log(f"CV R2  {name:22s} {m_:.3f} ± {s_:.3f}")
log(f"best params (all {p}): {best}")
rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, **best).fit(X, y)
gm = GeneraModel(X)

log("\n## Genera model: share of each genus' variance explained by the environment block")
r2e = 1 - gm.res.var(0) / X[:, GI].var(0)
Z = (gm.Epool - gm.Epool.mean(0)) / gm.Esd
std_coef = np.linalg.lstsq(np.c_[np.ones(n), Z], (X[:, GI] - X[:, GI].mean(0)) / X[:, GI].std(0), rcond=None)[0][1:]
tab_g = pd.DataFrame(std_coef.T, index=GEN, columns=[ENV_SHORT[e] for e in ENV]).round(2)
tab_g.insert(0, "R2_from_E", r2e.round(3))
log(tab_g.to_string())
tab_g.to_csv(os.path.join(OUT, "genera_given_environment.csv"))

# ------------------------------------------------------------------ main explanation
log(f"\n## Attributions (M = {M_MAIN} Monte-Carlo draws per coalition)")
vm = Game(rf, gm, "marginal", M_MAIN, SEED)
vc = Game(rf, gm, "conditional", M_MAIN, SEED + 1)
vd = Game(rf, gm, "causal", M_MAIN, SEED + 2)
allp = list(range(p))
res = {"SHAP-marginal": shapley(vm, allp)}
print("  marginal done", flush=True)
res["SHAP-conditional"] = shapley(vc, allp)
print("  conditional done", flush=True)
res["Causal SV"] = shapley(vd, allp)
print("  causal done", flush=True)
res["ASV env-first"] = asv(vc, EI, GI)
res["ASV genera-first"] = asv(vc, GI, EI)
res["path"] = paths(rf, gm, vc, M_MAIN, SEED + 3)

f0 = rf.predict(X)
for m_, v_ in [("SHAP-marginal", vm), ("SHAP-conditional", vc), ("Causal SV", vd)]:
    log(f"{m_:17s}: v(empty) = {v_([])[0]:.2f} (mean RF prediction {f0.mean():.2f}); "
        f"max efficiency gap {np.abs(res[m_].sum(1) - (f0 - v_([])[0])).max():.1e}")

tab = pd.DataFrame({m_: np.abs(res[m_]).mean(0) for m_ in METHODS}, index=LABEL)
direc = pd.DataFrame({m_: [pd.Series(X[:, j]).corr(pd.Series(res[m_][:, j]), method="spearman")
                           for j in range(p)] for m_ in METHODS}, index=LABEL)
log("\nmean |attribution| (FHB %p)")
log(tab.round(2).to_string())
log("\ndirection: Spearman rho(feature value, attribution)")
log(direc.round(2).to_string())

kept = (tab.loc[GEN, "ASV env-first"] / tab.loc[GEN, "ASV genera-first"]).rename("kept_env_first/genera_first")
kept_c = (tab.loc[GEN, "ASV env-first"] / tab.loc[GEN, "SHAP-conditional"]).rename("kept_vs_SHAP-conditional")
path = pd.Series(np.abs(res["path"]).mean(0), index=GEN, name="env credit via genus")
log("\nCredit each genus keeps when the environment explains first")
log(pd.concat([kept, kept_c, path], axis=1).round(3).to_string())
env_tot = {m_: tab.loc[[ENV_SHORT[e] for e in ENV], m_].sum() for m_ in METHODS}
gen_tot = {m_: tab.loc[GEN, m_].sum() for m_ in METHODS}
log("\nTotal mean |credit|: environment vs genera")
log(pd.DataFrame({"environment": env_tot, "genera": gen_tot}).round(2).to_string())

lo = X[:, FEATS.index("Alternaria")] < 7.2
split = pd.DataFrame({m_: [res[m_][lo, FEATS.index("Alternaria")].mean(),
                           res[m_][~lo, FEATS.index("Alternaria")].mean()] for m_ in METHODS},
                     index=[f"Alternaria < 7.2 (n={lo.sum()})", f"Alternaria >= 7.2 (n={(~lo).sum()})"])
log("\nAlternaria mean attribution by abundance side (FHB %p)")
log(split.round(2).to_string())

# ------------------------------------------------------------------ bootstrap
log(f"\n## Bootstrap ({N_BOOT} refits; ASVs and paths, M = 30)")
rng = np.random.default_rng(SEED)
boot = []
for b in range(N_BOOT):
    ib = rng.integers(0, n, n)
    mb = RandomForestRegressor(n_estimators=200, random_state=b, n_jobs=-1, **best).fit(X[ib], y[ib])
    gb = GeneraModel(X[ib])
    vb = Game(mb, gb, "conditional", 30, 100 + b)
    a1, a2 = asv(vb, EI, GI), asv(vb, GI, EI)
    pb = paths(mb, gb, vb, 30, 200 + b)
    for k, g in enumerate(GEN):
        j = NE + k
        boot.append(dict(b=b, genus=g, ASV_env_first=np.abs(a1[:, j]).mean(),
                         ASV_genera_first=np.abs(a2[:, j]).mean(),
                         kept=np.abs(a1[:, j]).mean() / np.abs(a2[:, j]).mean(),
                         path=np.abs(pb[:, k]).mean(),
                         rho_env_first=pd.Series(X[:, j]).corr(pd.Series(a1[:, j]), method="spearman"),
                         alt_low_side=a1[lo, j].mean() if g == "Alternaria" else np.nan,
                         alt_high_side=a1[~lo, j].mean() if g == "Alternaria" else np.nan))
    print(f"  bootstrap {b + 1}/{N_BOOT}", flush=True)
boot = pd.DataFrame(boot)
boot.to_csv(os.path.join(OUT, "bootstrap.csv"), index=False)
ci = boot.groupby("genus")[["ASV_env_first", "ASV_genera_first", "kept", "path", "rho_env_first"]] \
    .quantile([.025, .5, .975]).unstack().loc[GEN]
log(ci.round(3).to_string())
alt = boot[boot.genus == "Alternaria"][["alt_low_side", "alt_high_side"]].quantile([.025, .5, .975])
log("\nAlternaria env-first ASV by abundance side, bootstrap quantiles")
log(alt.round(2).to_string())

# ------------------------------------------------------------------ save
tab.round(4).to_csv(os.path.join(OUT, "mean_abs_attribution.csv"))
direc.round(4).to_csv(os.path.join(OUT, "direction_rho.csv"))
pd.concat([kept, kept_c, path], axis=1).round(4).to_csv(os.path.join(OUT, "kept_share_and_paths.csv"))
split.round(4).to_csv(os.path.join(OUT, "alternaria_split.csv"))
for m_ in METHODS:
    pd.DataFrame(res[m_], columns=LABEL).round(4).to_csv(
        os.path.join(OUT, f"values_{m_.replace(' ', '_')}.csv"), index=False)
pd.DataFrame(res["path"], columns=GEN).round(4).to_csv(os.path.join(OUT, "values_path.csv"), index=False)

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.7, 1]})
ax = axes[0]
w = .16
for k, m_ in enumerate(METHODS):
    ax.bar(np.arange(p) + (k - 2) * w, tab[m_], w * .92, color=MCOL[m_], label=m_)
ax.axvline(NE - .5, color="#52514e", lw=.8, ls=":")
ax.set_xticks(range(p), LABEL, rotation=25, ha="right")
ax.set_ylabel("mean |attribution| (FHB %p)")
ax.set_title("Attribution under five schemes (left of dotted line = environment block)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
q = boot.groupby("genus")["kept"].quantile([.025, .5, .975]).unstack().loc[GEN]
ax.barh(GEN, kept.loc[GEN] * 100, color=[COL[g] for g in GEN], height=.6)
ax.errorbar(q[.5] * 100, GEN, xerr=[(q[.5] - q[.025]) * 100, (q[.975] - q[.5]) * 100],
            fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
for yi, v in enumerate(kept.loc[GEN] * 100):
    ax.text(2, yi, f"{v:.0f}%", va="center", fontsize=8, color="white")
ax.axvline(100, color="#52514e", ls="--", lw=.8)
ax.set_xlabel("credit kept when environment explains first\n(ASV env-first / ASV genera-first, %; bootstrap 95% interval)")
ax.set_title("Own (non-environmental) share of each genus")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_attribution_and_kept_share.png"), dpi=200); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
for ax, g in zip(axes.ravel(), GEN):
    j = FEATS.index(g)
    ax.scatter(X[:, j], res["ASV genera-first"][:, j], s=14, color="#c9c8c2", label="genera-first")
    ax.scatter(X[:, j], res["ASV env-first"][:, j], s=20, color=COL[g], edgecolor="white", label="env-first")
    ax.axhline(0, color="#52514e", lw=.6, ls="--")
    ax.set_title(f"{g}: kept {kept[g] * 100:.0f}%, ρ(env-first) = {direc.loc[g, 'ASV env-first']:.2f}", fontsize=9)
    ax.set_xlabel(f"{g} (CLR)")
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("attribution (FHB %p)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Per-sample ASV: genera credited first (grey) vs environment credited first (colour)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_asv_dependence.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4))
pq = boot.groupby("genus")["path"].quantile([.025, .5, .975]).unstack().loc[GEN]
ax.barh(GEN, path.loc[GEN], color=[COL[g] for g in GEN], height=.6)
ax.errorbar(pq[.5], GEN, xerr=[pq[.5] - pq[.025], pq[.975] - pq[.5]], fmt="none", ecolor="#0b0b0b",
            capsize=3, lw=1)
ax.set_xlabel("mean |environment credit routed through genus| (FHB %p)")
ax.set_title("Path attribution: environment → genus → FHB")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_path_attribution.png"), dpi=200); plt.close(fig)

with open(os.path.join(OUT, "causal_shapley_weather_log.txt"), "w") as fh:
    fh.write("\n".join(report))
print(f"\nOutputs in {OUT}")
