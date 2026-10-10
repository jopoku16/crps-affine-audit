# Real-data validation protocol

Frozen before the first real-data source-compression run on 2026-10-08.

## Data

Use the `temp` dataset from CRAN package `ensemblepp` version 1.0-0. It contains 2,749 minimum-temperature observations and 11 corresponding NOAA GEFS reforecast members for Innsbruck Airport from 2000-01-02 through 2016-01-01. No rows are missing.

## Source and transformation

Use released CRPS code at commit `dddd3ea46a6199d2ed48019c7825e79fb242787e` without edits. For each row, score the forecast in degrees Celsius and after adding 273.15 to all ensemble members and the observation. The latter is the same physical forecast expressed in Kelvin.

## Configuration and pairing

Use all 11 members, 8 landmarks, and 5 compressed nodes. Use seed `20261008 + row_index` for both Celsius and Kelvin calls in each row. Evaluate the released compressed score, normalized pair correction, weighted empirical CRPS, and full fair CRPS. Score returned nodes and weights in float64; source compression remains float32.

## Checks and endpoints

The run is valid only if every call completes, weights are finite and nonnegative, weight sums differ from one by at most `2e-4`, public-function and independent released-score calculations differ by at most `2e-4` in 100 fixed audit rows, and the full fair score changes by at most `1e-10` under conversion.

Primary endpoint: paired absolute Celsius-to-Kelvin score change. Report the median, 95th percentile, maximum, and count above `0.01` degrees for each method. A method passes the practical invariance check only if its maximum change is at most `2e-4`.

Secondary endpoint: RMSE against the full 11-member fair CRPS, separately in Celsius and Kelvin. Report paired row-block bootstrap 95% percentile intervals for each alternative minus the released compressed method using 2,000 resamples and seed `20261009`.

Any failed source or numerical control remains a failure. Diagnosis may explain but not overwrite it.
