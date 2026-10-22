"""Target sensitivity: relative versus absolute error, and within-ACE-stratum signal.

The manuscript's grouped analysis shows that association/diagnostic features add
no independent predictive power for AREE. This script tests whether that
conclusion is specific to the relative-error target:

  1. Refit the same random forest on absolute error in probability points
     (`abs_error_pp`) and compare permutation importance with the AREE model.
  2. Within deciles of the observable ACE estimate (`logit_ace`), test the
     out-of-sample incremental R2 of the association/diagnostic group over a
     baseline of effect estimates plus sample characteristics.

The split, hyperparameters, and seed match `meta_train.py` (M1/M5 training,
M6/M7 test; 1,000 trees, depth 10, leaf 5; seed 20240625).

Outputs:
  results/raw/meta_target_sensitivity.json
  results/tables/meta_target_importance.csv
  results/tables/meta_target_within_stratum.csv
  results/compare/meta_target_sensitivity.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
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
EFFECT = ["logit_ace", "lpm_x_coef", "unadj_ace"]
SAMPLE = ["n", "y_mean", "x_var"]
ASSOC = [
    "logit_lpm_diff", "adj_unadj_diff", "z_coef_mag", "z_x_coef_mag",
    "z_x_r2", "y_pred_min", "y_pred_max", "y_pred_range", "y_model_r2",
]
GROUPS = {
    "effect_estimates": EFFECT,
    "association_diagnostics": ASSOC,
    "sample_characteristics": SAMPLE,
}
TARGETS = ["abs_bias_pct", "abs_error_pp"]


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    """Impute with training medians to avoid test-set leakage."""
    x_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians)
    x_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians)
    return x_train, x_test, medians


def fit_and_importance(x_train, y_train, x_test, y_test):
    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    metrics = {
        "r2_test": float(r2_score(y_test, pred)),
        "mae_test": float(mean_absolute_error(y_test, pred)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    imp = permutation_importance(
        rf, x_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1
    )
    raw = np.asarray(imp.importances_mean, dtype=float)
    positive = np.clip(raw, 0.0, None)
    total = positive.sum()
    share = positive / total if total > 0 else np.zeros_like(positive)
    rows = []
    for idx in np.argsort(raw)[::-1]:
        rows.append({
            "feature": FEATURES[idx],
            "importance": float(raw[idx]),
            "share": float(share[idx]),
            "rank": int(np.where(np.argsort(raw)[::-1] == idx)[0][0]) + 1,
        })
    group_importance = {
        name: float(sum(raw[FEATURES.index(f)] for f in cols))
        for name, cols in GROUPS.items()
    }
    group_share = {
        name: float(sum(share[FEATURES.index(f)] for f in cols))
        for name, cols in GROUPS.items()
    }
    return metrics, rows, group_importance, group_share


def within_stratum(test: pd.DataFrame, target: str, n_bins: int = 10,
                   n_splits: int = 20) -> list[dict]:
    """Out-of-sample incremental R2 of association features within ACE deciles.

    Within each decile of `logit_ace`, split 50/50. Fit OLS on one half with
    (effect + sample) and with (effect + sample + association), and score on
    the other half. Positive values mean the association group adds
    out-of-sample signal beyond the effect estimates and sample size.
    """
    frame = test.copy()
    frame["_bin"] = pd.qcut(frame["logit_ace"], n_bins, duplicates="drop",
                            labels=False)
    base_cols = EFFECT + SAMPLE
    full_cols = EFFECT + SAMPLE + ASSOC
    rng = np.random.default_rng(SEED)
    # Impute once on the test frame; the split is internal, so this is a
    # descriptive robustness check rather than a deployment model.
    x_all = frame[full_cols].replace([np.inf, -np.inf], np.nan)
    x_all = x_all.fillna(x_all.median())
    rows = []
    for bin_id, group in frame.groupby("_bin"):
        idx = group.index
        x_bin = x_all.loc[idx]
        y_bin = frame.loc[idx, target].to_numpy(dtype=float)
        n = len(y_bin)
        if n < 40:
            continue
        deltas = []
        base_scores = []
        full_scores = []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            half = n // 2
            tr, te = perm[:half], perm[half:]
            model_b = LinearRegression().fit(x_bin.iloc[tr][base_cols],
                                             y_bin[tr])
            model_f = LinearRegression().fit(x_bin.iloc[tr][full_cols],
                                             y_bin[tr])
            score_b = model_b.score(x_bin.iloc[te][base_cols], y_bin[te])
            score_f = model_f.score(x_bin.iloc[te][full_cols], y_bin[te])
            base_scores.append(float(score_b))
            full_scores.append(float(score_f))
            deltas.append(float(score_f - score_b))
        rows.append({
            "bin": int(bin_id),
            "n": int(n),
            "mean_ace": float(frame.loc[idx, "logit_ace"].mean()),
            "mean_r2_base": float(np.mean(base_scores)),
            "mean_r2_full": float(np.mean(full_scores)),
            "mean_delta_r2": float(np.mean(deltas)),
            "median_delta_r2": float(np.median(deltas)),
            "share_positive": float(np.mean(np.asarray(deltas) > 0)),
            "n_splits": int(n_splits),
        })
    return rows


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "meta_training_data.parquet")
    train = df[df["model_type"].isin(["M1", "M5"])].copy()
    test = df[df["model_type"].isin(["M6", "M7"])].copy()
    x_train, x_test, _ = prepare(train, test)

    payload = {
        "meta": {
            "seed": SEED,
            "train_models": ["M1", "M5"],
            "test_models": ["M6", "M7"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_repeats": 10,
            "features": FEATURES,
            "groups": GROUPS,
        },
        "targets": {},
        "within_stratum": {},
    }
    importance_rows = []
    stratum_rows = []
    for target in TARGETS:
        metrics, rows, group_importance, group_share = fit_and_importance(
            x_train, train[target].to_numpy(dtype=float),
            x_test, test[target].to_numpy(dtype=float),
        )
        payload["targets"][target] = {
            **metrics,
            "importance": rows,
            "group_importance": group_importance,
            "group_share": group_share,
        }
        for row in rows:
            importance_rows.append({"target": target, **row})
        print(f"\n=== {target} ===")
        print(f"R2={metrics['r2_test']:.4f} MAE={metrics['mae_test']:.4f} "
              f"RMSE={metrics['rmse_test']:.4f}")
        for row in rows[:6]:
            print(f"  {row['feature']:<18} {row['importance']:8.4f} "
                  f"share={row['share']:.3f}")
        print("  group importance:", group_importance)
        print("  group share:     ", group_share)

        strata = within_stratum(test, target)
        payload["within_stratum"][target] = {
            "n_bins": len(strata),
            "n_splits": 20,
            "mean_delta_r2": float(np.mean([r["mean_delta_r2"] for r in strata])),
            "median_delta_r2": float(np.median([r["mean_delta_r2"] for r in strata])),
            "share_bins_positive": float(np.mean([r["mean_delta_r2"] > 0 for r in strata])),
            "bins": strata,
        }
        for row in strata:
            stratum_rows.append({"target": target, **row})
        print("  within-stratum delta R2:",
              f"mean={payload['within_stratum'][target]['mean_delta_r2']:.4f}",
              f"median={payload['within_stratum'][target]['median_delta_r2']:.4f}",
              f"share_bins_positive={payload['within_stratum'][target]['share_bins_positive']:.2f}")

    (RAW / "meta_target_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(importance_rows).to_csv(
        TABLES / "meta_target_importance.csv", index=False
    )
    pd.DataFrame(stratum_rows).to_csv(
        TABLES / "meta_target_within_stratum.csv", index=False
    )

    lines = [
        "# Target sensitivity: relative versus absolute error",
        "",
        "Same random forest and split as the main analysis (M1/M5 training, "
        "M6/M7 test; 1,000 trees, depth 10, leaf 5; seed 20240625). "
        "Importance is the mean decrease in MSE under 10 permutations; "
        "share divides positive importance by its total.",
        "",
        "## Model performance",
        "",
        "| target | R2 | MAE | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        m = payload["targets"][target]
        lines.append(f"| {target} | {m['r2_test']:.4f} | {m['mae_test']:.4f} | "
                     f"{m['rmse_test']:.4f} |")
    lines += ["", "## Top-10 importance by target", "",
              "| rank | AREE feature | AREE importance | AREE share | "
              "absolute-error feature | abs. importance | abs. share |",
              "|---:|---|---:|---:|---|---:|---:|"]
    top = {t: payload["targets"][t]["importance"][:10] for t in TARGETS}
    for i in range(10):
        a = top["abs_bias_pct"][i]
        b = top["abs_error_pp"][i]
        lines.append(
            f"| {i + 1} | {a['feature']} | {a['importance']:.4f} | "
            f"{a['share']:.3f} | {b['feature']} | {b['importance']:.4f} | "
            f"{b['share']:.3f} |"
        )
    lines += ["", "## Group importance", "",
              "| group | AREE sum | AREE share | abs. sum | abs. share |",
              "|---|---:|---:|---:|---:|"]
    for name in GROUPS:
        a = payload["targets"]["abs_bias_pct"]
        b = payload["targets"]["abs_error_pp"]
        lines.append(
            f"| {name} | {a['group_importance'][name]:.4f} | "
            f"{a['group_share'][name]:.3f} | "
            f"{b['group_importance'][name]:.4f} | {b['group_share'][name]:.3f} |"
        )
    lines += ["", "## Within-ACE-decile incremental R2", "",
              "Each decile is split 50/50; OLS is fit on one half with "
              "(effect + sample) and with (effect + sample + association), "
              "and scored on the other half. Positive values mean the "
              "association group adds out-of-sample signal.", "",
              "| target | bins | mean delta R2 | median delta R2 | share of bins > 0 |",
              "|---|---:|---:|---:|---:|"]
    for target in TARGETS:
        w = payload["within_stratum"][target]
        lines.append(
            f"| {target} | {w['n_bins']} | {w['mean_delta_r2']:.4f} | "
            f"{w['median_delta_r2']:.4f} | {w['share_bins_positive']:.2f} |"
        )
    lines += ["", "Per-bin values are in `meta_target_within_stratum.csv`.", ""]
    (OUT / "meta_target_sensitivity.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nSaved {RAW / 'meta_target_sensitivity.json'}")
    print(f"Saved {TABLES / 'meta_target_importance.csv'}")
    print(f"Saved {TABLES / 'meta_target_within_stratum.csv'}")
    print(f"Saved {OUT / 'meta_target_sensitivity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
