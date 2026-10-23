"""Uplift / targeting layer + refutation pack for the Hillstrom RCT.

Called from src/analysis.py (the single entry point); merges two new
sections into outputs/results.json:

  uplift_targeting -- T-learner and X-learner uplift models for
      any-email vs no-email (outcome: conversion; spend secondary),
      evaluated with Qini curves / AUUC on out-of-fold scores, plus a
      policy table (email-all vs target-top-50% vs target-top-25%)
      estimated with the Qini policy-value estimator.
  refutation -- (a) placebo treatment: permuted assignment through the
      same PLR double-ML pipeline (effect must be ~0); (b) pseudo-
      outcome: pre-treatment history spend as the outcome with the real
      assignment (effect must be ~0).

Definitions used (Radcliffe-style Qini):
  For the top fraction phi of customers ranked by predicted uplift,
  with n_t / y_t the treated count / outcome sum inside the targeted
  set and n_c / y_c the control analogues,
      q(phi) = y_t - y_c * (n_t / n_c)
  is the estimated total incremental outcome from treating that set.
  The Qini curve plotted and integrated is q(phi) / N (incremental
  outcomes per customer on the file); the random-targeting baseline is
  the diagonal phi * ATE. AUUC = area under q(phi)/N; AUUC gain =
  AUUC - ATE/2.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import KFold

SEED = 42
N_SPLITS = 5


def _fit_mu(Xtr, ytr, kind):
    if kind == "clf":
        return HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    return HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr)


def _pred_mu(model, Xte, kind):
    return model.predict_proba(Xte)[:, 1] if kind == "clf" else model.predict(Xte)


def _crossfit_mu_arr(Xv, Wv, y, kind):
    """Out-of-fold mu1 = E[Y|X,W=1] and mu0 = E[Y|X,W=0] on raw arrays."""
    mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(Xv):
        for w, mu in ((1, mu1), (0, mu0)):
            m = _fit_mu(Xv[tr][Wv[tr] == w], y[tr][Wv[tr] == w], kind)
            mu[te] = _pred_mu(m, Xv[te], kind)
    return mu1, mu0


def crossfit_mu(X, W, y, kind):
    return _crossfit_mu_arr(X.values, W, y, kind)


def t_learner_scores(X, W, y, kind):
    """Out-of-fold uplift scores: mu1 - mu0, models fit per outer fold."""
    s = np.zeros(len(y))
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    Xv, Wv = X.values, W
    for tr, te in kf.split(Xv):
        m1 = _fit_mu(Xv[tr][Wv[tr] == 1], y[tr][Wv[tr] == 1], kind)
        m0 = _fit_mu(Xv[tr][Wv[tr] == 0], y[tr][Wv[tr] == 0], kind)
        s[te] = _pred_mu(m1, Xv[te], kind) - _pred_mu(m0, Xv[te], kind)
    return s


def x_learner_scores(X, W, y, kind):
    """Out-of-fold X-learner scores.

    Per outer fold: inner cross-fitted mu on the training part gives
    pseudo-outcomes D1 = y - mu0 (treated) and D0 = mu1 - y (control);
    tau1 / tau0 regressors are fit on those, then combined with the
    propensity e(x): tau = e * tau0 + (1 - e) * tau1.
    """
    s = np.zeros(len(y))
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    Xv, Wv = X.values, W
    for tr, te in kf.split(Xv):
        Xtr, Wtr, ytr = Xv[tr], Wv[tr], y[tr]
        mu1_in, mu0_in = _crossfit_mu_arr(Xtr, Wtr, ytr, kind)
        d = np.where(Wtr == 1, ytr - mu0_in, mu1_in - ytr)
        tau1 = HistGradientBoostingRegressor(random_state=SEED).fit(
            Xtr[Wtr == 1], d[Wtr == 1])
        tau0 = HistGradientBoostingRegressor(random_state=SEED).fit(
            Xtr[Wtr == 0], d[Wtr == 0])
        clf = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, Wtr)
        e_te = clf.predict_proba(Xv[te])[:, 1]
        s[te] = e_te * tau0.predict(Xv[te]) + (1 - e_te) * tau1.predict(Xv[te])
    return s


def qini_curve(y, W, scores, grid=None):
    """Per-capita Qini curve q(phi)/N on a fraction grid."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    order = np.argsort(-scores, kind="stable")
    ys, ws = y[order], W[order]
    N = len(y)
    q = np.zeros(len(grid))
    for i, phi in enumerate(grid):
        n = int(round(phi * N))
        if n == 0:
            continue
        wt, wc = ws[:n], ws[:n]
        n_t = int(wt.sum())
        n_c = n - n_t
        if n_t == 0 or n_c == 0:
            q[i] = q[i - 1] if i else 0.0
            continue
        y_t = ys[:n][wt == 1].sum()
        y_c = ys[:n][wc == 0].sum()
        q[i] = (y_t - y_c * (n_t / n_c)) / N
    return grid, q


def auuc_stats(y, W, scores):
    grid, q = qini_curve(y, W, scores)
    auuc = float(np.trapezoid(q, grid))
    ate = float(q[-1])
    return {"auuc": round(auuc, 8), "auuc_random": round(ate / 2, 8),
            "auuc_gain_over_random": round(auuc - ate / 2, 8),
            "ate_at_phi_1": round(ate, 8)}, grid, q


def policy_value(y, W, scores, phi):
    """Qini policy-value estimate for targeting the top-phi by score."""
    N = len(y)
    order = np.argsort(-scores, kind="stable")
    n = int(round(phi * N))
    idx = order[:n]
    wt, yt = W[idx], y[idx]
    n_t, n_c = int(wt.sum()), n - int(wt.sum())
    y_t = yt[wt == 1].sum()
    y_c = yt[wt == 0].sum()
    return float(y_t - y_c * (n_t / n_c))  # total incremental outcome


def run_uplift(df, X, W):
    """Returns the uplift_targeting results dict; saves the Qini figure."""
    out = {"outcome_primary": "conversion",
           "treatment": "any email vs no email",
           "evaluation": "Qini curve / AUUC on 5-fold cross-fitted (out-of-fold) scores"}
    curves = {}
    for outcome, kind in (("conversion", "clf"), ("spend", "reg")):
        y = df[outcome].values.astype(float)
        stats = {}
        for name, fn in (("t_learner", t_learner_scores),
                         ("x_learner", x_learner_scores)):
            s = fn(X, W, y, kind)
            st, grid, q = auuc_stats(y, W, s)
            stats[name] = st
            if outcome == "conversion":
                curves[name] = (grid, q, s)
        key = "learners" if outcome == "conversion" else "learners_spend_secondary"
        out[key] = stats

    gain_t = out["learners"]["t_learner"]["auuc_gain_over_random"]
    gain_x = out["learners"]["x_learner"]["auuc_gain_over_random"]
    chosen = "x_learner" if gain_x >= gain_t else "t_learner"
    out["chosen_learner_by_auuc"] = chosen

    scores = curves[chosen][2]
    y_conv = df["conversion"].values.astype(float)
    y_spend = df["spend"].values.astype(float)
    N = len(df)
    table = {}
    for label, phi in (("email_all", 1.0), ("target_top_50pct", 0.5),
                       ("target_top_25pct", 0.25)):
        inc_conv = policy_value(y_conv, W, scores, phi)
        inc_rev = policy_value(y_spend, W, scores, phi)
        sent = int(round(phi * N))
        table[label] = {
            "emails_sent": sent,
            "emails_avoided": N - sent,
            "incremental_conversions_total_est": round(inc_conv, 2),
            "incremental_conversions_per_1000_customers": round(1000 * inc_conv / N, 2),
            "incremental_conversions_per_1000_emailed": round(1000 * inc_conv / sent, 2),
            "incremental_revenue_per_1000_customers_usd": round(1000 * inc_rev / N, 2),
            "incremental_revenue_per_1000_emailed_usd": round(1000 * inc_rev / sent, 2),
        }
    out["policy_table"] = table
    out["policy_note"] = ("Policies rank customers by the chosen learner's "
                          "out-of-fold conversion-uplift score; incremental "
                          "conversions and revenue are estimated with the Qini "
                          "policy-value estimator (treated response minus "
                          "control response scaled to the treated count) "
                          "inside the targeted set.")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (grid, q, _) in curves.items():
        ax.plot(grid * 100, q * 100, label=f"{name.replace('_', '-')} (conversion uplift)")
    ate = curves[chosen][1][-1]
    ax.plot([0, 100], [0, ate * 100], "k--", lw=1, label="random targeting")
    ax.set_xlabel("% of customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("Incremental conversions per 100 customers")
    ax.set_title("Qini curves — email targeting, Hillstrom RCT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/qini_curve.png", dpi=150)
    return out


# --- refutation pack ---------------------------------------------------------

def _crossfit_preds_arr(X, y, w, kind):
    pred = np.zeros(len(y))
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    Xv = X.values
    for tr, te in kf.split(Xv):
        m = _fit_mu(Xv[tr], y[tr] if kind == "reg" else w[tr], kind)
        pred[te] = _pred_mu(m, Xv[te], kind)
    return pred


def dml_plr_general(X, y, w):
    """Same PLR double-ML estimator as src/analysis.py, parameterized."""
    l_hat = _crossfit_preds_arr(X, y, w, "reg")
    m_hat = _crossfit_preds_arr(X, y, w, "clf")
    w_res = w - m_hat
    theta = np.sum(w_res * (y - l_hat)) / np.sum(w_res * w)
    psi = w_res * (y - l_hat - theta * w)
    j = np.mean(w_res * w)
    se = np.sqrt(np.mean(psi ** 2) / j ** 2 / len(y))
    return {"estimate": round(float(theta), 6),
            "ci95": [round(float(theta - 1.96 * se), 6),
                     round(float(theta + 1.96 * se), 6)]}


def run_refutation(df, X, W):
    rng = np.random.default_rng(12345)
    w_placebo = rng.permutation(W)
    placebo = {outcome: dml_plr_general(X, df[outcome].values.astype(float),
                                        w_placebo.astype(float))
               for outcome in ("conversion", "spend")}
    pseudo = dml_plr_general(X, df["history"].values.astype(float), W)
    return {
        "placebo_treatment_permuted_assignment": placebo,
        "pseudo_outcome_pretreatment_history_spend": pseudo,
        "note": ("Both checks must be ~0: the placebo breaks the "
                 "treatment-outcome link while keeping the pipeline "
                 "identical, and history spend is pre-treatment, so the "
                 "real assignment cannot have caused it. Estimates are "
                 "reported as computed, whatever they are."),
    }
