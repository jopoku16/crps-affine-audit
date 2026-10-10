# Data and source provenance

This release does not include private or confidential data.

## Innsbruck temperature and precipitation

The Innsbruck forecasts come from `ensemblepp` version 1.0-0 on CRAN. The
package contains 12-hour minimum-temperature and precipitation observations
and 11 NOAA GEFS reforecast members. The package is distributed under GPL-2
or GPL-3. The file `real_data/extract_real_data.R` records the extraction.

- Package: https://cran.r-project.org/package=ensemblepp
- Version: 1.0-0
- Local analysis files: `temp.csv` and `rain.csv`

## Pacific Northwest surface temperature

The surface-temperature forecasts come from `ensembleBMA` version 5.1.8 on
CRAN. The package is distributed under GPL version 2 or later. The file
`real_data/extract_real_data.R` records the extraction.

- Package: https://cran.r-project.org/package=ensembleBMA
- Version: 5.1.8
- Local analysis file: `srft.csv`

## Audited source

The audited implementation is the public `ma921/CRPS` repository at commit
`dddd3ea46a6199d2ed48019c7825e79fb242787e`. It is not redistributed in this
release. Run `python real_data/fetch_pinned_source.py` to clone the repository,
check out the named commit, and verify the checked-out hash.

- Repository: https://github.com/ma921/CRPS
- Pinned tree: https://github.com/ma921/CRPS/tree/dddd3ea46a6199d2ed48019c7825e79fb242787e

The final publication associated with that repository is available from PMLR:
https://proceedings.mlr.press/v271/adachi25a.html
