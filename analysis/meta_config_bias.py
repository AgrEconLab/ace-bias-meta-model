"""Configuration-level bias and MSE decomposition (revision analysis).

For each of the 702 configurations, average the 20 Monte Carlo estimates and
decompose the mean squared error into bias^2 and sampling variance:

    mean_i (est_i - true)^2 = (mean_i est_i - true)^2 + mean_i (est_i - mean_i est_i)^2

Outputs:
  results/raw/meta_config_bias.json
  results/tables/meta_config_bias.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
TABLES = ROOT / "../results/tables"
SEED = 20240625

FEATURES = [
    "n", "y_mean", "x_var",
    "logit_ace", "logit_lpm_diff",
    "unadj_ace", "adj_unadj_diff",
    "z_coef_mag", "z_x_coef_mag", "z_x_r2",
    "y_pred_min", "y_pred_max", "y_pred_range",
    "lpm_x_coef", "y_model_r2",
]
GROUP_COLS = [
    "model_type", "case_type", "n", "lambda_xy",
    "vary_param_1_name", "vary_param_1_val",
    "vary_param_2_name", "vary_param_2_val",
]


def config_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(GROUP_COLS, dropna=False):
        est = g["logit_ace"].to_numpy(float)
        true = float(g["true_ace"].iloc[0])
        mean_est = float(np.mean(est))
        var = float(np.var(est, ddof=0))
        bias = mean_est - true
        mse = float(np.mean((est - true) ** 2))
        row = dict(zip(GROUP_COLS, key))
        row.update({
            "n_iter": int(len(g)),
            "true_ace": true,
            "mean_logit_ace": mean_est,
            "bias": bias,
            "bias_pct": bias / abs(true) * 100 if true != 0 else np.nan,
            "abs_bias_pct": abs(bias / abs(true) * 100) if true != 0 else np.nan,
            "variance": var,
            "mse": mse,
            "bias_sq": bias ** 2,
            "bias_sq_share": bias ** 2 / mse if mse > 0 else np.nan,
            "variance_share": var / mse if mse > 0 else np.nan,
            "mean_abs_error_pp": float(np.mean(np.abs(est - true))),
            "mean_abs_rel_error_pct": float(np.mean(np.abs((est - true) / true) * 100))
            if true != 0 else np.nan,
            "high_bias_rate": float(np.mean(np.abs((est - true) / true) > 0.05))
            if true != 0 else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame, by: str | None) -> pd.DataFrame:
    if by is None:
        groups = [("ALL", summary)]
    else:
        groups = [(str(k), g) for k, g in summary.groupby(by)]
    rows = []
    for name, g in groups:
        mse = g["mse"].mean()
        bs = g["bias_sq"].mean()
        var = g["variance"].mean()
        rows.append({
            by or "group": name,
            "n_config": len(g),
            "mean_bias": g["bias"].mean(),
            "mean_abs_bias": g["bias"].abs().mean(),
            "mean_bias_sq": bs,
            "mean_variance": var,
            "mean_mse": mse,
            "bias_sq_share": bs / mse if mse > 0 else np.nan,
            "variance_share": var / mse if mse > 0 else np.nan,
            "mean_abs_error_pp": g["mean_abs_error_pp"].mean(),
            "mean_abs_rel_error_pct": g["mean_abs_rel_error_pct"].mean(),
            "mean_high_bias_rate": g["high_bias_rate"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "meta_training_data.parquet")
    summary = config_summary(df)
    summary.to_csv(TABLES / "meta_config_bias.csv", index=False)

    overall = aggregate(summary, None)
    by_dag = aggregate(summary, "model_type")
    by_n = aggregate(summary, "n")
    by_case = aggregate(summary, "case_type")

    # Configuration-level feature means for the meta-model
    feat = df.groupby(GROUP_COLS, dropna=False)[[f for f in FEATURES if f != "n"]].mean().reset_index()
    merged = feat.merge(
        summary[GROUP_COLS + ["bias_pct", "abs_bias_pct", "mean_abs_rel_error_pct"]],
        on=GROUP_COLS, how="inner",
    )
    train_models = ["M1", "M5"]
    test_models = ["M6", "M7"]
    train = merged[merged["model_type"].isin(train_models)]
    test = merged[merged["model_type"].isin(test_models)]
    model_result = {"train_n": len(train), "test_n": len(test)}
    if len(train) > 10 and len(test) > 5:
        rf = RandomForestRegressor(
            n_estimators=1000, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED,
        )
        rf.fit(train[FEATURES], train["abs_bias_pct"])
        pred = rf.predict(test[FEATURES])
        model_result.update({
            "r2_test": float(r2_score(test["abs_bias_pct"], pred)),
            "mae_test": float(mean_absolute_error(test["abs_bias_pct"], pred)),
            "target": "configuration-level absolute bias (%)",
            "train_models": train_models,
            "test_models": test_models,
        })

    out = {
        "overall": overall.to_dict("records"),
        "by_dag": by_dag.to_dict("records"),
        "by_n": by_n.to_dict("records"),
        "by_case": by_case.to_dict("records"),
        "config_model": model_result,
    }
    (RAW / "meta_config_bias.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Overall:")
    print(overall.to_string(index=False))
    print("\nBy DAG:")
    print(by_dag.to_string(index=False))
    print("\nBy N:")
    print(by_n.to_string(index=False))
    print("\nConfiguration-level meta-model:")
    print(json.dumps(model_result, indent=2))
    print(f"\nSaved {TABLES / 'meta_config_bias.csv'}")
    print(f"Saved {RAW / 'meta_config_bias.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
