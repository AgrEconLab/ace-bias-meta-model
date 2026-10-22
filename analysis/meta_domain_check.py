"""Training-range (domain) check for the meta-model.

The external-validation section shows that the meta-model does not extrapolate
beyond its training range. This script turns that statement into an operational
check: for each new dataset, flag features outside the training range and
quantify the distance from the boundary.

Two rules are reported:
  hard: outside the observed training [min, max].
  soft: outside the training [0.1%, 99.9%] interval.

The public helper `check_domain(frame, reference)` returns a per-row flag and a
standardized distance, so the released package can warn users before applying
the meta-model.

Outputs:
  results/raw/meta_domain_check.json
  results/tables/meta_domain_check.csv
  results/tables/meta_domain_check_features.csv
  results/compare/meta_domain_check.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
TABLES = ROOT / "../results/tables"
OUT = ROOT / "../results/compare"

FEATURES = [
    "n", "y_mean", "x_var",
    "logit_ace", "logit_lpm_diff",
    "unadj_ace", "adj_unadj_diff",
    "z_coef_mag", "z_x_coef_mag", "z_x_r2",
    "y_pred_min", "y_pred_max", "y_pred_range",
    "lpm_x_coef", "y_model_r2",
]
EXTERNAL = {
    "Probit": "meta_ext_1_Probit.parquet",
    "ExtendedRange": "meta_ext_2_ExtendedRange.parquet",
    "UniformZ": "meta_ext_3_UniformZ.parquet",
}


def reference_stats(train: pd.DataFrame, features=None) -> dict:
    """Training min/max, 0.1-99.9% interval, mean, and SD per feature."""
    features = features or [f for f in FEATURES if f in train.columns]
    ref = {}
    for feature in features:
        s = pd.to_numeric(train[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        ref[feature] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "q001": float(s.quantile(0.001)),
            "q999": float(s.quantile(0.999)),
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=0)),
            "n": int(len(s)),
        }
    return ref


def check_domain(frame: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Flag rows outside the training range.

    Returns one row per input row with `hard_flag` (any feature outside
    [min, max]), `soft_flag` (any feature outside [q001, q999]), and
    `distance` (the largest standardized distance beyond the hard range).
    """
    features = [f for f in reference if f in frame.columns]
    out = pd.DataFrame(index=frame.index)
    hard = pd.Series(False, index=frame.index)
    soft = pd.Series(False, index=frame.index)
    distance = pd.Series(0.0, index=frame.index)
    for feature in features:
        x = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        stats = reference[feature]
        below_hard = x < stats["min"]
        above_hard = x > stats["max"]
        below_soft = x < stats["q001"]
        above_soft = x > stats["q999"]
        hard = hard | below_hard.fillna(False) | above_hard.fillna(False)
        soft = soft | below_soft.fillna(False) | above_soft.fillna(False)
        sd = stats["sd"] if stats["sd"] > 0 else np.nan
        if np.isfinite(sd):
            d = pd.concat([
                ((stats["min"] - x) / sd).clip(lower=0),
                ((x - stats["max"]) / sd).clip(lower=0),
            ], axis=1).max(axis=1)
            distance = np.maximum(distance, d.fillna(0.0))
    out["hard_flag"] = hard.astype(bool)
    out["soft_flag"] = soft.astype(bool)
    out["distance"] = distance.astype(float)
    return out


def summarise(frame: pd.DataFrame, reference: dict) -> dict:
    flags = check_domain(frame, reference)
    features = [f for f in reference if f in frame.columns]
    per_feature = {}
    for feature in features:
        x = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        stats = reference[feature]
        per_feature[feature] = {
            "hard_rate": float(((x < stats["min"]) | (x > stats["max"])).mean()),
            "soft_rate": float(((x < stats["q001"]) | (x > stats["q999"])).mean()),
        }
    return {
        "n": int(len(frame)),
        "features_checked": features,
        "missing_features": [f for f in FEATURES if f not in frame.columns],
        "hard_flag_rate": float(flags["hard_flag"].mean()),
        "soft_flag_rate": float(flags["soft_flag"].mean()),
        "median_distance": float(flags["distance"].median()),
        "p90_distance": float(flags["distance"].quantile(0.90)),
        "max_distance": float(flags["distance"].max()),
        "per_feature": per_feature,
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(RAW / "meta_training_data.parquet")
    train = train[train["model_type"].isin(["M1", "M5"])]
    reference = reference_stats(train)

    payload = {
        "meta": {
            "reference": "M1/M5 training set",
            "n_reference": int(len(train)),
            "hard_rule": "outside training [min, max]",
            "soft_rule": "outside training [0.1%, 99.9%]",
            "distance": "max standardized distance beyond the hard range (training SD units)",
        },
        "reference": reference,
        "datasets": {},
    }
    summary_rows = []
    feature_rows = []
    for name, filename in EXTERNAL.items():
        path = RAW / filename
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue
        frame = pd.read_parquet(path)
        result = summarise(frame, reference)
        payload["datasets"][name] = result
        summary_rows.append({
            "dataset": name,
            "n": result["n"],
            "hard_flag_rate": result["hard_flag_rate"],
            "soft_flag_rate": result["soft_flag_rate"],
            "median_distance": result["median_distance"],
            "p90_distance": result["p90_distance"],
            "max_distance": result["max_distance"],
        })
        for feature, values in result["per_feature"].items():
            feature_rows.append({"dataset": name, "feature": feature, **values})
        print(f"{name}: n={result['n']} hard={result['hard_flag_rate']:.3f} "
              f"soft={result['soft_flag_rate']:.3f} "
              f"median_dist={result['median_distance']:.2f}")

    (RAW / "meta_domain_check.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(
        TABLES / "meta_domain_check.csv", index=False
    )
    pd.DataFrame(feature_rows).to_csv(
        TABLES / "meta_domain_check_features.csv", index=False
    )

    lines = [
        "# Training-range check",
        "",
        "The reference is the M1/M5 training set. The hard rule flags a row if "
        "any feature lies outside the training [min, max]; the soft rule uses "
        "the training [0.1%, 99.9%] interval. Distance is the largest "
        "standardized distance beyond the hard range (training SD units).",
        "",
        "| dataset | n | hard flag | soft flag | median distance | p90 distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['n']} | {row['hard_flag_rate']:.3f} | "
            f"{row['soft_flag_rate']:.3f} | {row['median_distance']:.2f} | "
            f"{row['p90_distance']:.2f} |"
        )
    lines += [
        "",
        "Per-feature violation rates are in "
        "`meta_domain_check_features.csv`; the full reference ranges are in "
        "`meta_domain_check.json`.",
        "",
    ]
    (OUT / "meta_domain_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {RAW / 'meta_domain_check.json'}")
    print(f"Saved {TABLES / 'meta_domain_check.csv'}")
    print(f"Saved {TABLES / 'meta_domain_check_features.csv'}")
    print(f"Saved {OUT / 'meta_domain_check.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
