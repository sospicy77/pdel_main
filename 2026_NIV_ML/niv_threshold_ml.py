"""
NIV toxin occurrence classification across multiple thresholds.

For each threshold t in {0.00, 0.02, ..., 0.20}, the binary outcome is
    y = 1 if NIV > t else 0
and several ML classifiers are tuned (Optuna / TPE) and evaluated with
*nested* cross-validation so that the reported AUC / Brier score are not
optimistically biased by hyperparameter tuning:

  outer loop : repeated stratified K-fold (performance estimation)
  inner loop : stratified K-fold inside each outer-training set (tuning)

After evaluation, every model is re-tuned on the full data set (more trials)
and saved as the final model for that threshold.

Usage:
    python niv_threshold_ml.py --data data/NIV_toxin_data.xlsx --out results
    python niv_threshold_ml.py --objective brier      # tune on Brier score
    python niv_threshold_ml.py --quick                # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from niv_preprocess import LogColumns

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET = "NIV (Y)"
FEATURES = [
    "FHB_incidence_predicted",
    "temp_before_hd",
    "rhum_before_hd",
    "prcp_before_hd",
    "temp_after_flower",
    "rhum_after_flower",
    "prcp_after_flower",
]
PRCP_IDX = [FEATURES.index("prcp_before_hd"), FEATURES.index("prcp_after_flower")]
THRESHOLDS = np.round(np.arange(0.0, 0.2001, 0.02), 2)


# --------------------------------------------------------------------------
# Preprocessing helpers
# --------------------------------------------------------------------------
def scaled(model, log_prcp: bool):
    """Pipeline: (optional log1p of precipitation) -> standardise -> model."""
    steps = []
    if log_prcp:
        steps.append(("logp", LogColumns(cols=tuple(PRCP_IDX))))
    steps += [("sc", StandardScaler()), ("clf", model)]
    return Pipeline(steps)


def cw(trial):
    return trial.suggest_categorical("class_weight", [None, "balanced"])


# --------------------------------------------------------------------------
# Search spaces  (trial -> unfitted estimator)
# --------------------------------------------------------------------------
def build_logreg(trial, seed):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95) if penalty == "elasticnet" else None
    m = LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True),
        l1_ratio={"l1": 1.0, "l2": 0.0, "elasticnet": l1_ratio}[penalty],
        solver="saga",
        class_weight=cw(trial),
        max_iter=5000,
        random_state=seed,
    )
    return scaled(m, trial.suggest_categorical("log_prcp", [False, True]))


def build_svm(trial, seed):
    m = SVC(
        C=trial.suggest_float("C", 1e-2, 1e3, log=True),
        gamma=trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        kernel="rbf",
        class_weight=cw(trial),
        probability=True,
        random_state=seed,
    )
    return scaled(m, trial.suggest_categorical("log_prcp", [False, True]))


def build_knn(trial, seed):
    m = KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_int("p", 1, 2),
    )
    return scaled(m, trial.suggest_categorical("log_prcp", [False, True]))


def build_mlp(trial, seed):
    n_layers = trial.suggest_int("n_layers", 1, 2)
    width = trial.suggest_int("width", 4, 32, log=True)
    m = MLPClassifier(
        hidden_layer_sizes=(width,) * n_layers,
        alpha=trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
        solver="lbfgs",
        max_iter=500,
        random_state=seed,
    )
    return scaled(m, trial.suggest_categorical("log_prcp", [False, True]))


def _forest_params(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
        max_depth=trial.suggest_categorical("max_depth", [2, 3, 4, 6, 8, None]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 16),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
    )


def build_rf(trial, seed):
    p = _forest_params(trial)
    return RandomForestClassifier(
        **p,
        bootstrap=True,
        class_weight=trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),
        random_state=seed,
        n_jobs=1,
    )


def build_et(trial, seed):
    p = _forest_params(trial)
    return ExtraTreesClassifier(**p, class_weight=cw(trial), random_state=seed, n_jobs=1)


def build_xgb(trial, seed):
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 1, 6),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def build_lgbm(trial, seed):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 2, 16),
        max_depth=trial.suggest_int("max_depth", 1, 6),
        min_child_samples=trial.suggest_int("min_child_samples", 2, 20),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        class_weight=cw(trial),
        min_split_gain=0.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


# Slow learners (forests, MLP) get a smaller Optuna budget; see trial_budget().
SLOW_MODELS = {"MLP", "RandomForest", "ExtraTrees"}


def trial_budget(name, n_trials, slow_factor):
    return max(5, int(round(n_trials * (slow_factor if name in SLOW_MODELS else 1.0))))


MODELS = {
    "LogisticRegression": build_logreg,
    "SVM_RBF": build_svm,
    "KNN": build_knn,
    "MLP": build_mlp,
    "RandomForest": build_rf,
    "ExtraTrees": build_et,
    "XGBoost": build_xgb,
    "LightGBM": build_lgbm,
}


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
def cv_score(est, X, y, cv, objective):
    """Pooled out-of-fold score inside the inner CV (higher is better)."""
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = clone(est).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    if objective == "auc":
        return roc_auc_score(y, oof)
    return -brier_score_loss(y, oof)


def tune(name, X, y, n_trials, inner_k, objective, seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # also inside worker processes
    warnings.filterwarnings("ignore")
    builder = MODELS[name]
    cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)

    def obj(trial):
        try:
            return cv_score(builder(trial, seed), X, y, cv, objective)
        except Exception:  # pragma: no cover - degenerate configs
            return float("-inf") if objective == "auc" else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=min(15, n_trials // 3), multivariate=True
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = builder(optuna.trial.FixedTrial(study.best_params), seed)
    return best, study.best_params, study.best_value


# --------------------------------------------------------------------------
# Nested CV for one (threshold, model)
# --------------------------------------------------------------------------
def nested_cv(name, X, y, args, thr):
    outer = RepeatedStratifiedKFold(
        n_splits=args.outer_k, n_repeats=args.outer_repeats, random_state=args.seed
    )
    n = len(y)
    oof = np.full((args.outer_repeats, n), np.nan)
    for i, (tr, te) in enumerate(outer.split(X, y)):
        rep = i // args.outer_k
        est, _, _ = tune(
            name, X[tr], y[tr], trial_budget(name, args.n_trials, args.slow_factor), args.inner_k, args.objective, args.seed + i
        )
        est.fit(X[tr], y[tr])
        oof[rep, te] = est.predict_proba(X[te])[:, 1]

    aucs = np.array([roc_auc_score(y, oof[r]) for r in range(args.outer_repeats)])
    briers = np.array([brier_score_loss(y, oof[r]) for r in range(args.outer_repeats)])
    prev = y.mean()
    brier_ref = prev * (1 - prev)  # Brier of a constant "prevalence" forecast
    return {
        "threshold": thr,
        "model": name,
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "AUC_mean": aucs.mean(),
        "AUC_sd": aucs.std(ddof=1) if len(aucs) > 1 else 0.0,
        "Brier_mean": briers.mean(),
        "Brier_sd": briers.std(ddof=1) if len(briers) > 1 else 0.0,
        "Brier_ref": brier_ref,
        "BSS_mean": 1 - briers.mean() / brier_ref,
    }, oof


def run_threshold(thr, X, niv, args, out):
    y = (niv > thr).astype(int)
    t0 = time.time()
    rows, oofs = [], {}
    for name in args.models:
        res, oof = nested_cv(name, X, y, args, thr)
        rows.append(res)
        oofs[name] = oof
        print(
            f"[t={thr:.2f}] {name:<18} AUC={res['AUC_mean']:.3f}±{res['AUC_sd']:.3f} "
            f"Brier={res['Brier_mean']:.4f} BSS={res['BSS_mean']:.3f}",
            flush=True,
        )

    # ---- final models: re-tune on all data with more trials -------------
    finals = {}
    for name in args.models:
        est, params, inner_val = tune(
            name, X, y, trial_budget(name, args.final_trials, args.slow_factor), args.inner_k, args.objective, args.seed
        )
        est.fit(X, y)
        finals[name] = {"params": params, "inner_cv_score": inner_val}
        joblib.dump(est, out / "models" / f"thr_{thr:.2f}_{name}.joblib")

    np.savez_compressed(out / "oof" / f"oof_thr_{thr:.2f}.npz", y=y, **oofs)
    print(f"[t={thr:.2f}] done in {time.time() - t0:.0f}s", flush=True)
    return rows, {f"{thr:.2f}": finals}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def make_plots(res, best, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, g in res.groupby("model"):
        g = g.sort_values("threshold")
        axes[0].errorbar(g.threshold, g.AUC_mean, yerr=g.AUC_sd, marker="o", ms=4,
                         capsize=2, label=name)
        axes[1].plot(g.threshold, g.Brier_mean, marker="o", ms=4, label=name)
    g = res.drop_duplicates("threshold").sort_values("threshold")
    axes[1].plot(g.threshold, g.Brier_ref, "k--", lw=1.2, label="Reference (prevalence)")
    axes[0].axhline(0.5, color="grey", lw=0.8, ls=":")
    axes[0].set(xlabel="NIV threshold", ylabel="Nested-CV AUC", title="AUC by threshold")
    axes[1].set(xlabel="NIV threshold", ylabel="Nested-CV Brier score (lower = better)",
                title="Brier score by threshold")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "performance_by_threshold.png", dpi=150)
    plt.close(fig)

    # ROC + calibration for best model at each threshold
    k = len(best)
    ncol = 4
    nrow = int(np.ceil(k / ncol))
    for kind in ("roc", "calibration"):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, (_, r) in zip(axes, best.iterrows()):
            d = np.load(out / "oof" / f"oof_thr_{r.threshold:.2f}.npz")
            y, p = d["y"], d[r.model].mean(axis=0)  # mean OOF prob across repeats
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, lw=2)
                ax.plot([0, 1], [0, 1], "k:", lw=0.8)
                ax.set(xlabel="1 - Specificity", ylabel="Sensitivity")
            else:
                pt, pp = calibration_curve(y, p, n_bins=5, strategy="quantile")
                ax.plot(pp, pt, "o-")
                ax.plot([0, 1], [0, 1], "k:", lw=0.8)
                ax.set(xlabel="Predicted probability", ylabel="Observed fraction",
                       xlim=(0, 1), ylim=(0, 1))
            ax.set_title(f"t={r.threshold:.2f} | {r.model}\nAUC={r.AUC_mean:.3f}, "
                         f"Brier={r.Brier_mean:.3f}", fontsize=9)
        for ax in axes[k:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"best_model_{kind}.png", dpi=130)
        plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--data", default=str(here / "data" / "NIV_toxin_data.xlsx"))
    ap.add_argument("--out", default=str(here / "results"))
    ap.add_argument("--objective", choices=["auc", "brier"], default="auc",
                    help="metric optimised during hyperparameter tuning")
    ap.add_argument("--outer-k", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=3)
    ap.add_argument("--inner-k", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=60, help="Optuna trials per outer fold")
    ap.add_argument("--slow-factor", type=float, default=0.4,
                    help="trial-budget multiplier for slow models (RF, ExtraTrees, MLP)")
    ap.add_argument("--final-trials", type=int, default=150, help="Optuna trials for final model")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.outer_repeats, args.n_trials, args.final_trials = 1, 6, 6

    out = Path(args.out)
    for sub in ("", "models", "oof"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.data)
    X = df[FEATURES].to_numpy(float)
    niv = df[TARGET].to_numpy(float)

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(run_threshold)(round(t, 2), X, niv, args, out) for t in args.thresholds
    )
    rows = [r for rs, _ in results for r in rs]
    finals = {}
    for _, f in results:
        finals.update(f)

    res = pd.DataFrame(rows).sort_values(["threshold", "AUC_mean"], ascending=[True, False])
    res.to_csv(out / "nested_cv_results.csv", index=False, float_format="%.4f")

    # best model per threshold: highest AUC; ties (within 0.005) -> lower Brier
    def pick(g):
        top = g[g.AUC_mean >= g.AUC_mean.max() - 0.005]
        return top.sort_values("Brier_mean").iloc[0]

    best = pd.DataFrame([pick(g) for _, g in res.groupby("threshold")]).reset_index(drop=True)
    best["final_params"] = [
        json.dumps(finals[f"{r.threshold:.2f}"][r.model]["params"]) for _, r in best.iterrows()
    ]
    best.to_csv(out / "best_model_per_threshold.csv", index=False, float_format="%.4f")

    best_brier = res.loc[res.groupby("threshold").Brier_mean.idxmin()]
    best_brier.to_csv(out / "best_brier_model_per_threshold.csv", index=False,
                      float_format="%.4f")

    with open(out / "final_hyperparameters.json", "w") as f:
        json.dump(finals, f, indent=2, default=str)
    with open(out / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    make_plots(res, best, out)
    print("\n=== Best model per threshold (nested CV) ===")
    print(best.drop(columns="final_params").to_string(index=False, float_format="%.3f"))
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
