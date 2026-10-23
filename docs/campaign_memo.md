# Campaign memo — what the Hillstrom experiment says to do next

**Question.** The 2008 Hillstrom experiment randomized 64,000 customers to
a men's email, a women's email, or no email. What should the next campaign
do differently: which creative, to whom, and how much volume?

All numbers below are computed in this repo (`outputs/results.json`,
produced by `src/analysis.py` + `src/uplift.py`). Effects are
intent-to-treat / double-ML estimates of any email vs no email unless an
arm is named; both estimators agree to two decimals.

## Recommendation

1. **Lead with the men's creative.** Per customer emailed, the men's
   email produced +0.68 pp conversion [0.50, 0.86] and +$0.77 spend
   [0.49, 1.05] vs control — roughly double the women's email
   (+0.31 pp [0.15, 0.47]; +$0.42 [0.17, 0.68]). On this customer file
   and season, if only one creative ships, ship the men's. Keep the
   women's creative in test rotation only with a pre-declared read;
   do not split volume evenly by default.
2. **Email the whole eligible file for a one-shot campaign; do not buy
   ML targeting yet.** Every pre-treatment segment has a positive
   conversion point estimate, and the uplift models do not rank
   customers reliably: on out-of-fold Qini/AUUC, the X-learner is
   within ~1% of random targeting (gain +0.00002 over a random-targeting
   area of 0.00165), the T-learner ranks *worse* than random
   (gain −0.00011), and for spend both learners rank worse than random.
   A targeting policy built on these scores would be process theater.
3. **If send volume must be cut (list-health pressure), cut by coarse
   segments, not by model score.** The only model-ranked policy with a
   measured efficiency gain is the top quartile by X-learner score:
   4.10 incremental conversions per 1,000 emailed vs 3.31 for
   email-all (+24% per send), but it keeps only ~31% of the total
   incremental conversions (1.02 vs 3.31 per 1,000 customers on the
   file) and ~26% of the revenue per 1,000 on file ($102 vs $398).
   Segment cuts with direct evidence: returning customers
   (spend effect +$0.35, CI [−0.00, 0.69]) and the 5–8-month recency
   band (spend +$0.32, CI [−0.11, 0.74]) are the weakest reads and the
   first places to trim. Protect the strongest: high history-spend
   customers (+0.64 pp conversion [0.35, 0.93]) and new customers
   (+0.63 pp [0.46, 0.81]; largest spend effect, +$0.85 [0.57, 1.12]).

## Budget translation (one campaign, whole file)

| Policy | Emails sent | Incremental conversions per 1,000 on file | Incremental revenue per 1,000 on file |
|---|---|---|---|
| Email all | 64,000 | 3.31 | $398 |
| Top 50% by uplift score | 32,000 | 1.70 | $187 |
| Top 25% by uplift score | 16,000 | 1.02 | $102 |

(Per-1,000-on-file figures average over all 64,000 customers including
those not emailed; the previously reported +4.96 conversions and +$597
per 1,000 *emailed* is the same email-all result on the emailed
denominator.) Spend is revenue, not profit — the dataset has no cost or
margin data, so these are gross figures against a near-zero marginal
cost per email.

## What we do not know (unmeasured in this dataset)

- **Unsubscribe and spam-complaint rates.** Nothing here measures list
  damage. A one-shot email-all recommendation does **not** extend to
  "email everyone every campaign": fatigue compounds, and this dataset
  contains a single send.
- **Anything past two weeks.** No repeat-purchase, retention, or
  lifetime-value outcomes. It is possible the emails mostly pulled
  purchases forward in time; a two-week window cannot rule that out.
- **Margin and fulfillment cost.** Revenue effects could look different
  net of the product mix each creative drives.
- **Why the men's creative won.** The arms differ in merchandise
  featured; the file's response may reflect the assortment or the
  season (spring 2008), not a durable creative law. One experiment,
  one retailer, 2008.

## Next campaign design (so the next read is better than this one)

- **Persistent 5% holdout.** Reserve 5% of the file, randomly drawn
  once and excluded from *all* campaign email across the season, not
  re-randomized per send. This is what turns the next dataset into a
  long-run measurement: cumulative effect of the program, fatigue, and
  90-day repeat purchase — none of which the 2008 design can answer.
- **Primary metric declared before launch:** 14-day conversion,
  reported as incremental conversions per 1,000 assigned. Secondary:
  incremental revenue per 1,000 assigned, unsubscribe rate, complaint
  rate. No metric gets promoted after the data is in.
- **Measure the harms this dataset lacks:** unsubscribes and complaints
  recorded per arm and per send, joined back to the assignment file.
- **Stratify the randomization** by history-spend tercile × new vs
  returning, so the segment reads that drove this memo's trimming
  advice come out powered instead of post hoc.
- **Analysis pre-committed:** ITT primary; double ML as the adjusted
  cross-check; the placebo and pseudo-outcome refutations from this
  repo run as standard (in this dataset both read ~0, as they must:
  permuted-assignment conversion −0.008 pp [−0.163, +0.148]; history
  spend as outcome +$0.19 [−$0.17, +$0.55]).
