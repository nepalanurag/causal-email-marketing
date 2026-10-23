# Data source

Hillstrom "MineThatData E-Mail Analytics / Data Mining Challenge" dataset
(Kevin Hillstrom, MineThatData, 2008): a randomized email experiment on
64,000 customers of an online retailer.

Original distribution filename:
`Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`

Downloaded from the public GitHub mirror:
https://github.com/nharrison917/Hillstrom_analysis/blob/master/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Cross-checked against a second independent mirror
(https://github.com/duyh1612/hillstrom-uplift-rct/blob/main/data/hillstrom_raw.csv):
the two files parse to identical data frames.

Stored here as `hillstrom.csv`.

## Verification (computed on download, 2026-10-22)

- Rows: 64,000
- Columns: recency, history_segment, history, mens, womens, zip_code,
  newbie, channel, segment, visit, conversion, spend
  (the brief's "new" covariate is the `newbie` column in the actual data)
- Treatment arms (column `segment`):
  - Womens E-Mail: 21,387
  - Mens E-Mail: 21,307
  - No E-Mail (control): 21,306
- Total conversions: 578; total visits: 9,394; total spend: $67,258.13
