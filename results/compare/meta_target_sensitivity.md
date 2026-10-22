# Target sensitivity: relative versus absolute error

Same random forest and split as the main analysis (M1/M5 training, M6/M7 test; 1,000 trees, depth 10, leaf 5; seed 20240625). Importance is the mean decrease in MSE under 10 permutations; share divides positive importance by its total.

## Model performance

| target | R2 | MAE | RMSE |
|---|---:|---:|---:|
| abs_bias_pct | 0.5506 | 63.1398 | 118.4962 |
| abs_error_pp | 0.6567 | 0.0124 | 0.0163 |

## Top-10 importance by target

| rank | AREE feature | AREE importance | AREE share | absolute-error feature | abs. importance | abs. share |
|---:|---|---:|---:|---|---:|---:|
| 1 | logit_ace | 0.3446 | 0.550 | lpm_x_coef | 0.4112 | 0.491 |
| 2 | lpm_x_coef | 0.1713 | 0.274 | logit_ace | 0.2028 | 0.242 |
| 3 | unadj_ace | 0.0651 | 0.104 | adj_unadj_diff | 0.1265 | 0.151 |
| 4 | n | 0.0399 | 0.064 | n | 0.0606 | 0.072 |
| 5 | logit_lpm_diff | 0.0039 | 0.006 | x_var | 0.0167 | 0.020 |
| 6 | y_mean | 0.0009 | 0.001 | unadj_ace | 0.0149 | 0.018 |
| 7 | z_x_coef_mag | 0.0005 | 0.001 | z_x_coef_mag | 0.0018 | 0.002 |
| 8 | y_pred_max | -0.0003 | 0.000 | logit_lpm_diff | 0.0017 | 0.002 |
| 9 | y_pred_range | -0.0012 | 0.000 | y_mean | 0.0010 | 0.001 |
| 10 | x_var | -0.0014 | 0.000 | y_pred_max | -0.0004 | 0.000 |

## Group importance

| group | AREE sum | AREE share | abs. sum | abs. share |
|---|---:|---:|---:|---:|
| effect_estimates | 0.5809 | 0.928 | 0.6290 | 0.751 |
| association_diagnostics | -0.0306 | 0.007 | 0.1171 | 0.155 |
| sample_characteristics | 0.0394 | 0.065 | 0.0783 | 0.094 |

## Within-ACE-decile incremental R2

Each decile is split 50/50; OLS is fit on one half with (effect + sample) and with (effect + sample + association), and scored on the other half. Positive values mean the association group adds out-of-sample signal.

| target | bins | mean delta R2 | median delta R2 | share of bins > 0 |
|---|---:|---:|---:|---:|
| abs_bias_pct | 10 | -0.0335 | -0.0486 | 0.10 |
| abs_error_pp | 10 | -0.0259 | -0.0241 | 0.10 |

Per-bin values are in `meta_target_within_stratum.csv`.
