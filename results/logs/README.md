# Run logs supporting manuscript numbers

These logs are included because a few reported metrics are printed by the
scripts but are not separately persisted in JSON/CSV. They are evidence of a
run, not a substitute for the result files.

- `reproduce_all.log` — full cold-start run (`reproduce_all.sh`), exit 0.
  - Random 80/20 split: `Overall R2: 0.5786, MAE: 53.68pp`.
  - Reverse held-out (M6/M7 → M1/M5): `RF R2 = 0.4742 | MAE = 52.89pp`.
  - Baseline MAE: `113.75pp` (mean) and `146.91pp` (zero).
- `reproduce_extensions.log` — all extension analyses (13 scripts), all rc=0.
- `extensions_run_summary.tsv` — per-script rc and elapsed time for the 13
  extension analyses, including target sensitivity, leave-one-DAG-out CV,
  training-range check, and classifier-threshold sensitivity.
- `ext_meta_target_sensitivity.log`, `ext_meta_lodo_cv.log`,
  `ext_meta_domain_check.log`, `ext_meta_classifier_threshold.log` — console
  output for the four internal-review analyses.
- `red_rule_2x2.log` — RED/GREEN operating characteristics on held-out M6/M7
  (`red_rule_2x2.py`); the same numbers are persisted in
  `results/tables/red_rule_2x2.csv`.
- `meta_multicov.log` — multi-covariate extension console output.
  - Single-covariate model applied to multi-covariate data: `R2 = 0.5512`,
    `MAE = 74.14pp`; per-DAG M9 0.6273, M8 0.4862, M10 0.4697.
  - Combined training: M9 0.8534, M10 0.7145, M8 0.6990; single held-out 0.6519.
  - Two-input tree: M9 0.6375, M10 0.3885, M8 0.3527.
  - Feature importance: logit_ace 0.4437, lpm_x_coef 0.3989, n 0.1195,
    x_var 0.0365, unadj_ace 0.0355.

All other manuscript numbers are persisted in `results/raw/*.json`,
`results/tables/*.csv`, or `results/figures/*`.
