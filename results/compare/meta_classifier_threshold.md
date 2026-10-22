# High-error classification at alternative thresholds

Logistic regression on the same 15 features, with the binary target `AREE > threshold`. The 5% threshold is the manuscript's original high-error indicator.

| regime | threshold | positive share (train) | positive share (test) | accuracy | precision | recall | kappa | balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary_heldout | 5% | 0.939 | 0.955 | 0.955 | 0.955 | 1.000 | 0.000 | 0.500 | 0.739 |
| primary_heldout | 25% | 0.711 | 0.792 | 0.816 | 0.825 | 0.973 | 0.255 | 0.596 | 0.782 |
| primary_heldout | 50% | 0.515 | 0.637 | 0.719 | 0.778 | 0.782 | 0.392 | 0.696 | 0.787 |
| random_stratified | 5% | 0.939 | 0.942 | 0.942 | 0.942 | 1.000 | 0.000 | 0.500 | 0.722 |
| random_stratified | 25% | 0.734 | 0.725 | 0.764 | 0.777 | 0.946 | 0.278 | 0.614 | 0.764 |
| random_stratified | 50% | 0.555 | 0.543 | 0.702 | 0.709 | 0.765 | 0.395 | 0.696 | 0.781 |

Confusion matrices are in `meta_classifier_threshold.json`.
