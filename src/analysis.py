"""Causal analysis of the Hillstrom email marketing RCT.

Estimates, in order:
  1. ITT (difference in means) per email arm vs control, and any-email vs
     control, for conversion and spend -- the randomized benchmark.
  2. Double ML (partially linear model, 5-fold cross-fitting, gradient
     boosting nuisances) ATE of any-email vs no-email, plus an AIPW
     cross-check.
  3. CATE by pre-treatment segments (within-segment diff-in-means, valid
     under randomization): history-spend groups, recency bands, new vs
     returning.
  4. E-value for the any-email conversion effect.
  5. Business translation: effect per 1,000 customers emailed.

Run:  .venv/bin/python src/analysis.py
Writes outputs/results.json and outputs/cate_by_segment.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import KFold

SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv("data/hillstrom.csv")

# --- data checks (the runnable check) -------------------------------------
assert len(df) == 64000, len(df)
arms = df["segment"].value_counts()
assert set(arms.index) == {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
assert arms.between(21000, 22000).all(), arms
print("DATA CHECK OK: 64,000 rows; arms:", arms.to_dict())

df["any_email"] = (df["segment"] != "No E-Mail").astype(int)
X = pd.get_dummies(
    df[["recency", "history", "mens", "womens", "newbie",
        "history_segment", "zip_code", "channel"]],
    columns=["history_segment", "zip_code", "channel"], drop_first=True,
).astype(float)


def diff_ci(y, t):
    """Difference in means with 95% CI (Welch)."""
    a, b = y[t == 1], y[t == 0]
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - 1.96 * se, d + 1.96 * se, se


def pack(d, lo, hi):
    return {"estimate": round(float(d), 6), "ci95": [round(float(lo), 6), round(float(hi), 6)]}


# --- 1. ITT ----------------------------------------------------------------
itt = {}
for name, label in [("mens_email", "Mens E-Mail"), ("womens_email", "Womens E-Mail"),
                    ("any_email", None)]:
    if label is None:
        sub, t = df, df["any_email"]
    else:
        sub = df[(df["segment"] == label) | (df["segment"] == "No E-Mail")]
        t = (sub["segment"] == label).astype(int)
    itt[name] = {
        "n_treated": int(t.sum()), "n_control": int((1 - t).sum()),
        "conversion": pack(*diff_ci(sub["conversion"].values, t.values)[:3]),
        "spend": pack(*diff_ci(sub["spend"].values, t.values)[:3]),
        "control_conversion_rate": float(sub.loc[t == 0, "conversion"].mean()),
        "treated_conversion_rate": float(sub.loc[t == 1, "conversion"].mean()),
    }

# --- 2. Double ML (PLR, cross-fitted) + AIPW cross-check -------------------
W = df["any_email"].values.astype(float)


def crossfit_preds(y, model_fn):
    pred = np.zeros(len(df))
    kf = KFold(5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict_proba(X.iloc[te])[:, 1] if hasattr(m, "predict_proba") else m.predict(X.iloc[te])
    return pred


def dml_plr(y):
    """Partially linear DML with cross-fitted l(X)=E[Y|X], m(X)=E[W|X]."""
    l_hat = crossfit_preds(y, lambda: HistGradientBoostingRegressor(random_state=SEED))
    m_hat = crossfit_preds(W, lambda: HistGradientBoostingClassifier(random_state=SEED))
    w_res = W - m_hat
    theta = np.sum(w_res * (y - l_hat)) / np.sum(w_res * W)
    psi = w_res * (y - l_hat - theta * W)
    j = np.mean(w_res * W)
    se = np.sqrt(np.mean(psi ** 2) / j ** 2 / len(y))
    return theta, theta - 1.96 * se, theta + 1.96 * se


def aipw(y):
    kf = KFold(5, shuffle=True, random_state=SEED)
    mu1, mu0, e = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for tr, te in kf.split(X):
        for w, mu in [(1, mu1), (0, mu0)]:
            m = HistGradientBoostingRegressor(random_state=SEED)
            m.fit(X.iloc[tr][W[tr] == w], y[tr][W[tr] == w])
            mu[te] = m.predict(X.iloc[te])
        clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(X.iloc[tr], W[tr])
        e[te] = clf.predict_proba(X.iloc[te])[:, 1]
    e = np.clip(e, 0.01, 0.99)
    score = mu1 - mu0 + W * (y - mu1) / e - (1 - W) * (y - mu0) / (1 - e)
    theta, se = score.mean(), score.std(ddof=1) / np.sqrt(len(y))
    return theta, theta - 1.96 * se, theta + 1.96 * se


dml, aipw_res = {}, {}
for outcome in ["conversion", "spend"]:
    y = df[outcome].values.astype(float)
    dml[outcome] = pack(*dml_plr(y))
    aipw_res[outcome] = pack(*aipw(y))

# --- 3. CATE by pre-treatment segments -------------------------------------
seg_defs = {}
hist_pos = df["history"]
edges = np.quantile(hist_pos, [1 / 3, 2 / 3])
seg_defs["history_spend"] = pd.cut(
    df["history"], [-np.inf, edges[0], edges[1], np.inf],
    labels=[f"low (<${edges[0]:.0f})", f"mid (${edges[0]:.0f}-${edges[1]:.0f})",
            f"high (>${edges[1]:.0f})"])
seg_defs["recency_months"] = pd.cut(df["recency"], [0, 4, 8, 12],
                                    labels=["1-4", "5-8", "9-12"])
seg_defs["customer_type"] = df["newbie"].map({1: "new", 0: "returning"})

cate = {}
for fam, labels in seg_defs.items():
    cate[fam] = {}
    for lab in labels.dropna().unique():
        sub = df[labels == lab]
        t = sub["any_email"].values
        c = diff_ci(sub["conversion"].values, t)
        s = diff_ci(sub["spend"].values, t)
        cate[fam][str(lab)] = {"n": int(len(sub)),
                               "conversion": pack(*c[:3]), "spend": pack(*s[:3])}

best_family, best_seg, best_val = None, None, -np.inf
for fam, segs in cate.items():
    for lab, r in segs.items():
        if r["conversion"]["estimate"] > best_val:
            best_family, best_seg, best_val = fam, lab, r["conversion"]["estimate"]

# --- 4. E-value for the any-email conversion effect ------------------------
p1 = itt["any_email"]["treated_conversion_rate"]
p0 = itt["any_email"]["control_conversion_rate"]
rr = p1 / p0
e_value = rr + np.sqrt(rr * (rr - 1))
se_log = np.sqrt(1 / (p1 * itt["any_email"]["n_treated"]) - 1 / itt["any_email"]["n_treated"]
                 + 1 / (p0 * itt["any_email"]["n_control"]) - 1 / itt["any_email"]["n_control"])
rr_lo = np.exp(np.log(rr) - 1.96 * se_log)
e_value_ci = rr_lo + np.sqrt(rr_lo * (rr_lo - 1))

# --- 5. Business translation (per 1,000 customers emailed) -----------------
conv_itt = itt["any_email"]["conversion"]["estimate"]
spend_itt = itt["any_email"]["spend"]["estimate"]
business = {
    "incremental_conversions_per_1000_emailed": round(conv_itt * 1000, 2),
    "incremental_conversions_per_1000_ci95": [round(itt["any_email"]["conversion"]["ci95"][0] * 1000, 2),
                                              round(itt["any_email"]["conversion"]["ci95"][1] * 1000, 2)],
    "incremental_revenue_per_1000_emailed_usd": round(spend_itt * 1000, 2),
    "incremental_revenue_per_1000_ci95_usd": [round(itt["any_email"]["spend"]["ci95"][0] * 1000, 2),
                                              round(itt["any_email"]["spend"]["ci95"][1] * 1000, 2)],
    "margin_note": ("`spend` is revenue, not profit; the dataset has no cost data. "
                    "Net gain = incremental revenue x the retailer's gross margin "
                    "(e.g. $" + f"{spend_itt * 1000 * 0.5:,.0f} per 1,000 emailed at an "
                    "illustrative 50% margin), against a near-zero marginal cost per email sent."),
}

results = {
    "dataset": {"n": int(len(df)), "arms": {k: int(v) for k, v in arms.items()},
                "control_conversion_rate": round(p0, 6),
                "control_mean_spend": round(float(df.loc[df["any_email"] == 0, "spend"].mean()), 4)},
    "itt_vs_control": itt,
    "dml_any_email_vs_control": dml,
    "aipw_any_email_vs_control": aipw_res,
    "cate_by_segment": cate,
    "largest_conversion_effect_segment": {"family": best_family, "segment": best_seg,
                                          "conversion_ate": round(float(best_val), 6)},
    "e_value_conversion": {"risk_ratio": round(float(rr), 4), "e_value_point": round(float(e_value), 3),
                           "e_value_ci_limit": round(float(e_value_ci), 3),
                           "caveat": ("Data are from a randomized trial, so unmeasured "
                                      "confounding is ruled out by design; the E-value is "
                                      "reported as the benchmark an observational analysis "
                                      "of the same effect would have to survive.")},
    "business_translation": business,
}
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- figure: CATE on conversion by segment ---------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
titles = {"history_spend": "History spend", "recency_months": "Recency (months)",
          "customer_type": "Customer type"}
for ax, (fam, segs) in zip(axes, cate.items()):
    labs = list(segs.keys())
    est = [segs[l]["conversion"]["estimate"] * 100 for l in labs]
    lo = [segs[l]["conversion"]["ci95"][0] * 100 for l in labs]
    hi = [segs[l]["conversion"]["ci95"][1] * 100 for l in labs]
    ypos = np.arange(len(labs))[::-1]
    ax.errorbar(est, ypos, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(ypos, labs, fontsize=8)
    ax.set_title(titles[fam], fontsize=10)
axes[0].set_xlabel("Conversion effect (percentage points, 95% CI)")
fig.tight_layout()
fig.savefig("outputs/cate_by_segment.png", dpi=150)

# --- console summary ---------------------------------------------------------
print(json.dumps(results, indent=2))
print("\nWrote outputs/results.json and outputs/cate_by_segment.png")
