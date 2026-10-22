# Negative-effect check of the fixed risk-zone rule

- targeted grid: 126 configs x 20 = 2520 datasets
- lambda_xy in {-0.05, -0.10, -0.30}; one path strength (0.30)

## Overall

- median |error|: 0.0160 pp
- median relative error: 67.2%
- high-error rate (>5% relative): 94.6%

## By fixed-rule zone

| zone | n | median |error| (pp) | median rel. error | high-error rate | mean lpm_x_coef |
|---|---:|---:|---:|---:|---:|
| RED | 1462 | 0.0198 | 51.3% | 93.6% | -0.062 |
| GREEN | 373 | 0.0281 | 185.7% | 99.7% | 0.021 |
| YELLOW-A | 675 | 0.0071 | 53.1% | 93.6% | -0.014 |
| YELLOW-B | 10 | 0.1359 | 1042.1% | 100.0% | 0.127 |

## Interpretation

If the RED zone contains many protective effects with low absolute error,
the rule is not sign-symmetric and should be restricted to positive effects.
