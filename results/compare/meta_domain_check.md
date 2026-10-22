# Training-range check

The reference is the M1/M5 training set. The hard rule flags a row if any feature lies outside the training [min, max]; the soft rule uses the training [0.1%, 99.9%] interval. Distance is the largest standardized distance beyond the hard range (training SD units).

| dataset | n | hard flag | soft flag | median distance | p90 distance |
|---|---:|---:|---:|---:|---:|
| Probit | 10800 | 0.014 | 0.083 | 0.00 | 0.00 |
| ExtendedRange | 2400 | 1.000 | 1.000 | 2.40 | 8.33 |
| UniformZ | 3600 | 0.006 | 0.072 | 0.00 | 0.00 |

Per-feature violation rates are in `meta_domain_check_features.csv`; the full reference ranges are in `meta_domain_check.json`.
