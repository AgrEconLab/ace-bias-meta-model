"""Leave-one-DAG-out cross-validation across all seven DAG families.

The primary evaluation trains on M1/M5 and tests on M6/M7. This script reports
the symmetric alternative: for each DAG, train on the other six and test on the
held-out one. It answers the concern that the primary split is asymmetric and
that the test set contains structurally higher error rates.

Same features, hyperparameters, and seed as `meta_train.py`.

Outputs:
  results/raw/meta_lodo_cv.json
  results/tables/meta_lodo_cv.csv
  results/compare/meta_lodo_cv.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
TABLES = ROOT / "../results/tables"
OUT = ROOT / "../results/compare"
SEED = 20240625

FEATURES = [
    "n", "y_mean", "x_var",
    "logit_ace", "logit_lpm_diff",
    "unadj_ace", "adj_unadj_diff",
    "z_coef_mag", "z_x_coef_mag", "z_x_r2",
    "y_pred_min", "y_pred_max", "y_pred_range",
    "lpm_x_coef", "y_model_r2",
]
TARGET = "abs_bias_pct"
DAGS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    x_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians)
    x_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians)
    return x_train, x_test


def fit_eval(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    x_train, x_test = prepare(train, test)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)
    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "meta_training_data.parquet")

    folds = []
    for held in DAGS:
        train = df[df["model_type"] != held]
        test = df[df["model_type"] == held]
        result = fit_eval(train, test)
        folds.append({"held_out": held, **result})
        print(f"held {held}: n_train={result['n_train']:5d} "
              f"n_test={result['n_test']:4d} R2={result['r2']:.4f} "
              f"MAE={result['mae']:.2f}")

    primary = fit_eval(
        df[df["model_type"].isin(["M1", "M5"])],
        df[df["model_type"].isin(["M6", "M7"])],
    )
    payload = {
        "meta": {
            "seed": SEED,
            "target": TARGET,
            "features": FEATURES,
            "n_estimators": 1000,
            "max_depth": 10,
            "min_samples_leaf": 5,
        },
        "folds": folds,
        "lodo_mean_r2": float(np.mean([f["r2"] for f in folds])),
        "lodo_mean_mae": float(np.mean([f["mae"] for f in folds])),
        "lodo_min_r2": float(np.min([f["r2"] for f in folds])),
        "lodo_max_r2": float(np.max([f["r2"] for f in folds])),
        "primary_split": primary,
    }
    (RAW / "meta_lodo_cv.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(folds).to_csv(TABLES / "meta_lodo_cv.csv", index=False)

    lines = [
        "# Leave-one-DAG-out cross-validation",
        "",
        "For each DAG, the random forest is trained on the other six DAGs and "
        "tested on the held-out one. Features, hyperparameters, and seed are "
        "the same as the main analysis (1,000 trees, depth 10, leaf 5; "
        "seed 20240625).",
        "",
        "| held-out DAG | n train | n test | R2 | MAE | RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        lines.append(
            f"| {f['held_out']} | {f['n_train']} | {f['n_test']} | "
            f"{f['r2']:.4f} | {f['mae']:.2f} | {f['rmse']:.2f} |"
        )
    lines += [
        "",
        f"Mean R2 = {payload['lodo_mean_r2']:.4f}; mean MAE = "
        f"{payload['lodo_mean_mae']:.2f}; range "
        f"{payload['lodo_min_r2']:.4f} to {payload['lodo_max_r2']:.4f}.",
        "",
        "Primary split for comparison (M1/M5 -> M6/M7): "
        f"R2 = {primary['r2']:.4f}, MAE = {primary['mae']:.2f}, "
        f"n_train = {primary['n_train']}, n_test = {primary['n_test']}.",
        "",
    ]
    (OUT / "meta_lodo_cv.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"LODO mean R2={payload['lodo_mean_r2']:.4f} "
          f"MAE={payload['lodo_mean_mae']:.2f}")
    print(f"Primary split R2={primary['r2']:.4f} MAE={primary['mae']:.2f}")
    print(f"Saved {RAW / 'meta_lodo_cv.json'}")
    print(f"Saved {TABLES / 'meta_lodo_cv.csv'}")
    print(f"Saved {OUT / 'meta_lodo_cv.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
