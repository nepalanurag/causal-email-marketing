# Causal inference on a real email marketing experiment (Hillstrom RCT)

Double machine learning applied to commercial data: the Hillstrom
MineThatData email experiment — 64,000 customers randomized to a men's
email, a women's email, or no email, with two weeks of visit, conversion,
and spend outcomes. The question a marketing team actually asks: *what did
the emails cause, per customer, and for whom did they work?*

Data provenance and verification: [data/SOURCE.md](data/SOURCE.md).
Full numeric output: [outputs/results.json](outputs/results.json);
run log: [RUN_LOG.txt](RUN_LOG.txt).

## Design

- **Treatment arms:** Mens E-Mail (n = 21,307), Womens E-Mail (n = 21,387),
  No E-Mail control (n = 21,306).
- **Outcomes:** `conversion` (bought within two weeks), `spend` ($).
  Control baseline: 0.57% conversion, $0.65 mean spend — most customers
  buy nothing, so effects live in the tail.
- **Covariates (all pre-treatment):** recency, history spend and segment,
  mens, womens, zip code, newbie, channel.

## Methods

1. **ITT benchmark.** Difference in means per arm vs control with 95% CIs.
   In a randomized trial this is unbiased by construction; it is the
   number every fancier method must reproduce.
2. **Double ML.** Partially linear model for any-email vs no-email,
   5-fold cross-fitting, gradient-boosting nuisance models for
   E[Y|X] and the propensity E[W|X] (sklearn `HistGradientBoosting`).
   An AIPW estimator with the same cross-fitting is run as a cross-check.
3. **CATE by segment.** Within-segment difference in means (valid under
   randomization) across pre-treatment segments: history-spend terciles,
   recency bands, new vs returning customers.
4. **E-value.** Sensitivity of the conversion effect to unmeasured
   confounding, reported with its randomized-design caveat below.
5. **Uplift / targeting.** T-learner and X-learner uplift models
   (gradient-boosting bases, 5-fold cross-fitting, out-of-fold scores),
   evaluated with Qini curves / AUUC against random targeting, and a
   policy table estimated with the Qini policy-value estimator
   (`src/uplift.py`).
6. **Refutation.** The same double-ML pipeline run on a permuted
   (placebo) assignment and on a pre-treatment pseudo-outcome
   (history spend); both must read ~0.

## Headline results

| Estimand | Conversion effect | Spend effect |
|---|---|---|
| ITT, men's email vs control | +0.68 pp [0.50, 0.86] | +$0.77 [0.49, 1.05] |
| ITT, women's email vs control | +0.31 pp [0.15, 0.47] | +$0.42 [0.17, 0.68] |
| ITT, any email vs control | +0.50 pp [0.35, 0.64] | +$0.60 [0.38, 0.82] |
| Double ML, any email | +0.50 pp [0.36, 0.64] | +$0.60 [0.38, 0.82] |
| AIPW cross-check, any email | +0.50 pp [0.36, 0.64] | +$0.59 [0.37, 0.81] |

Double ML and AIPW land on top of the ITT benchmark to two decimal
places. That is the expected result under randomization — covariate
adjustment has no confounding to remove — and it is the point of running
them here: the same pipeline, validated where the truth is known by
design, is the one to trust on observational marketing data where it is
not.

**Where the effect concentrates.** The largest conversion effect is in
the high history-spend tercile (>$255 past-year spend): +0.64 pp
[0.35, 0.93]. New customers respond almost as strongly on conversion
(+0.63 pp [0.46, 0.81]) and show the largest spend effect of any segment,
+$0.85 per customer [0.57, 1.12]. The men's email outperforms the
women's email on this customer file on both outcomes.

![CATE by segment](outputs/cate_by_segment.png)

**Sensitivity.** The any-email conversion risk ratio is 1.87, giving an
E-value of 3.14 (2.43 for the confidence limit nearest the null): an
unmeasured confounder associated with both treatment and conversion by
a risk ratio of ~3.1 each could explain the effect away. In this dataset
that is a benchmark, not a worry — assignment was randomized, so
unmeasured confounding is ruled out by design.

**Business translation.** Per 1,000 customers emailed: **+4.96
incremental conversions [3.55, 6.36] and +$597 incremental revenue
[$376, $817]**. `spend` is revenue, not profit; the dataset has no cost
data, so net gain is the revenue figure times the retailer's gross
margin (≈ $298 per 1,000 at an illustrative 50% margin), against a
near-zero marginal cost per email sent.

## Targeting: does ranking customers help?

Barely — and that is a computed result, not an impression. T-learner
and X-learner uplift models for any-email vs no-email were scored
out-of-fold and evaluated with Qini curves. The Qini value at fraction
φ is the estimated incremental conversions per customer on the file
from targeting the top φ by predicted uplift; random targeting is the
diagonal, so AUUC for random targeting is half the φ = 1 value
(0.00165 for conversion).

| Outcome | Learner | AUUC | Gain over random |
|---|---|---|---|
| Conversion | T-learner | 0.00154 | −0.00011 |
| Conversion | X-learner | 0.00168 | +0.00002 |
| Spend | T-learner | 0.18007 | −0.01899 |
| Spend | X-learner | 0.15451 | −0.04455 |

The X-learner (chosen for the policy table, as the better of the two)
beats random targeting by ~1% of the curve area on conversion; the
T-learner and both spend rankings are worse than random. With 578
conversions in 64,000 customers, individual-level effect heterogeneity
is mostly noise here, even though coarse segment effects (above) are
real. The policy consequences, estimated with the Qini policy-value
estimator on the out-of-fold X-learner ranking:

| Policy | Emails sent | Emails avoided | Incr. conversions per 1,000 on file | Incr. revenue per 1,000 on file |
|---|---|---|---|---|
| Email all | 64,000 | 0 | 3.31 | $398 |
| Target top 50% | 32,000 | 32,000 | 1.70 | $187 |
| Target top 25% | 16,000 | 48,000 | 1.02 | $102 |

Targeting the top quartile raises efficiency per send (4.10 vs 3.31
incremental conversions per 1,000 emailed) but keeps only ~31% of the
total incremental conversions. Since a segment-level read shows no
group with a negative effect and email's marginal cost is near zero,
email-all is the volume-maximizing policy for a one-shot campaign;
targeting only pays if send volume itself carries a cost (fatigue,
list health) — which this dataset does not measure. The full
recommendation is in the [campaign memo](docs/campaign_memo.md).

![Qini curves](outputs/qini_curve.png)

## Refutation checks

Same double-ML pipeline, two ways it must find nothing:

- **Placebo treatment** (assignment randomly permuted): conversion
  −0.008 pp [−0.163, +0.148]; spend −$0.09 [−$0.35, +$0.16]. Both
  cover zero and are an order of magnitude below the real effects.
- **Pseudo-outcome** (pre-treatment history spend as the outcome,
  real assignment): +$0.19 [−$0.17, +$0.55], on a covariate with a
  mean of ~$242. The assignment cannot have caused pre-treatment
  spend, and the pipeline correctly finds that it didn't.

## Limitations

- One retailer, one campaign, spring 2008; absolute dollar effects do
  not transport to other lists or years.
- Spend is zero-inflated and heavy-tailed — a small number of large
  orders drives the spend estimates, hence the wide CIs. Segment-level
  spend effects for returning customers and the 5–8 month recency band
  have CIs that cross zero.
- Segment CATEs are subgroup average effects from stratified
  randomization checks, not individual-level effect estimates; no
  multiplicity adjustment across segments.
- Two-week outcome window only; no long-term or unsubscribe/harm
  outcomes are in the data, so "email everyone" is not a supported
  conclusion — only the average incremental effect is.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/analysis.py   # writes outputs/ and prints results
```

## Resume bullet

> Estimated the causal effect of email marketing on 64,000 randomized
> customers using cross-fitted double machine learning, validated
> against the randomized ITT benchmark; emailing produced +5.0
> incremental conversions and +$597 revenue per 1,000 customers
> (95% CI $376–$817), concentrated in high-history-spend and new
> customers.
