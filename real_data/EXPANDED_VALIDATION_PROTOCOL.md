# Expanded real-data and correction validation protocol

Prespecified locally before the first expanded run on 2026-10-08. This is not described as a public preregistration.

## Data

1. Innsbruck: `ensemblepp` 1.0-0 `temp`, 2,749 ordered cases, one station, 11 NOAA GEFS reforecast members, 2000-01-02 to 2016-01-01. Original values are Celsius; add 273.15 for Kelvin.
2. Pacific Northwest: `ensembleBMA` 5.1.8 `srft`, 36,826 cases, 969 stations and 52 available dates, eight ensemble members, January-February 2004. Original values are Kelvin; subtract 273.15 for Celsius.

Rows with a nonfinite observation or member would be excluded and counted. The supplied files currently have none.

## Unchanged source and candidate correction

Use the released source at commit `dddd3ea46a6199d2ed48019c7825e79fb242787e` without edits. Source compression stays in float32; score calculations use float64.

The candidate correction first subtracts the minimum ensemble member from every member before calling the unchanged compressor. This anchor makes compression input unchanged by a common translation and keeps the compression nodes nonnegative. Score the returned weighted distribution with weighted empirical CRPS against the observation after subtracting the same anchor. This is called anchored weighted compression. It is a software correction, not an unbiased-population-CRPS claim.

## Full-data external validation

Use eight landmarks and five nodes for Innsbruck, and six landmarks and four nodes for the Pacific Northwest. Pair the two units with the same seed `20261008 + row_index`. Compare released compressed, normalized pair, weighted empirical, anchored weighted compression, and full fair CRPS.

Controls require no source failures; finite, nonnegative weights; maximum weight-sum error at most `2e-4`; maximum full-fair unit-conversion error at most `1e-10`; and maximum anchored-correction unit-conversion error at most `2e-4`.

Primary endpoints are median, 95th percentile, maximum, and proportion of absolute unit-conversion score changes above 0.01 degrees. For Innsbruck, use a moving-block bootstrap over consecutive rows with block lengths 7, 14, and 30. For the Pacific Northwest, resample the 52 dates as clusters. Use 2,000 bootstrap replicates and seed `20261009`.

## Seed and budget sensitivity

Use 400 evenly spaced rows from each dataset, ten additional seeds per row, and paired units. Innsbruck configurations are `(5,3)`, `(8,3)`, `(8,5)`, `(10,5)`, and `(10,8)` for landmarks and nodes. Pacific Northwest configurations are `(4,3)`, `(6,3)`, `(6,4)`, `(7,4)`, and `(7,6)`. Report failures and the median, 95th percentile, and maximum change across all row-seed pairs for released, weighted empirical, and anchored weighted compression. No failed configuration is replaced.

## Equal-budget baselines and timing

On the same 400-row subsets and the middle configuration for each dataset, compare released compressed, raw compressed weighted empirical, anchored weighted compression, a seeded simple random subsample of the same node count scored by fair CRPS, and equally spaced order statistics of the same node count scored by empirical CRPS. Use ten paired seeds for stochastic methods. Compare each score with full-ensemble fair CRPS. Report RMSE and measured wall time. Timing is descriptive and includes Python-loop overhead.

## Real forecast-method ranking

Use Innsbruck rows 365 onward. At each row use only the previous 365 rows to construct four 11-member forecasts: raw ensemble; rolling mean-bias correction; rolling affine correction; and rolling observation climatology. For the affine correction, select spread multiplier from `{0.6,0.8,1.0,1.2,1.4}` by the lowest mean fair CRPS on the preceding window after mean-bias correction. The climatology uses eleven equally spaced empirical quantiles of the preceding observations.

Score all four candidates in Celsius and Kelvin with released compression, anchored weighted compression, and full fair CRPS. Use the same seed for each paired unit call. The primary ranking endpoint is the number and proportion of cases whose winning candidate changes after unit conversion. Ties use the first candidate in the fixed order above. Report moving-block-bootstrap 95% intervals for block lengths 7, 14, and 30, using 2,000 replicates and seed `20261010`.

## Interpretation

Full fair CRPS is an uncompressed reference, not population truth. Equal-budget results measure approximation of that reference. A locally prespecified protocol is reported as prespecified, not publicly preregistered. The second dataset improves external validity but does not identify the unreleased final-paper implementation.
