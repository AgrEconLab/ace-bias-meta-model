# Leave-one-DAG-out cross-validation

For each DAG, the random forest is trained on the other six DAGs and tested on the held-out one. Features, hyperparameters, and seed are the same as the main analysis (1,000 trees, depth 10, leaf 5; seed 20240625).

| held-out DAG | n train | n test | R2 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| M1 | 10800 | 3240 | 0.4928 | 59.43 | 118.27 |
| M2 | 12960 | 1080 | 0.5841 | 58.12 | 97.95 |
| M3 | 12960 | 1080 | 0.5671 | 60.58 | 107.07 |
| M4 | 12960 | 1080 | 0.5138 | 56.95 | 104.12 |
| M5 | 10800 | 3240 | 0.3846 | 48.01 | 80.71 |
| M6 | 10800 | 3240 | 0.6130 | 61.91 | 113.95 |
| M7 | 12960 | 1080 | 0.4985 | 56.52 | 105.73 |

Mean R2 = 0.5220; mean MAE = 57.36; range 0.3846 to 0.6130.

Primary split for comparison (M1/M5 -> M6/M7): R2 = 0.5506, MAE = 63.14, n_train = 6480, n_test = 4320.
