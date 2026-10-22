# Grouped feature importance and group removal (reviewer M4)

- full model: 15 features, held-out R2=0.5506, MAE=63.14

## Grouped permutation importance (R2 drop when the whole group is permuted)

| group | features | mean R2 drop | SD |
|---|---:|---:|---:|
| E_effect | 3 | 0.9103 | 0.0204 |
| A_association_diagnostics | 9 | -0.0093 | 0.0034 |
| S_sample | 3 | 0.0443 | 0.0045 |

## Group removal (refit without the group)

| group | remaining features | R2 | MAE |
|---|---:|---:|---:|
| E_effect | 12 | -0.0157 | 106.91 |
| A_association_diagnostics | 6 | 0.5515 | 62.69 |
| S_sample | 12 | 0.5639 | 67.02 |

## Full model

- R2=0.5506, MAE=63.14
