# Correlated-covariate sensitivity (M8/M10)

| model | rho | n | mean abs error (pp) | mean max|Z coef| | corr(max|Z|, error) | 5-fold CV R2 | CV MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| M10 | 0.0 | 40 | 0.0080 | 0.276 | 0.281 | 0.321 | 0.0030 |
| M10 | 0.6 | 40 | 0.0077 | 0.292 | 0.460 | 0.615 | 0.0027 |
| M8 | 0.0 | 40 | 0.0071 | 0.269 | 0.175 | 0.275 | 0.0039 |
| M8 | 0.6 | 40 | 0.0071 | 0.271 | 0.147 | 0.258 | 0.0041 |

Targeted sensitivity only; not the full multi-covariate extension.
