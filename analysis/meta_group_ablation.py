"""Grouped feature importance and group-removal robustness (reviewer M4).

Groups:
  E = effect estimates: logit_ace, lpm_x_coef, unadj_ace
  A = association/diagnostics: logit_lpm_diff, adj_unadj_diff, z_coef_mag,
      z_x_coef_mag, z_x_r2, y_pred_min, y_pred_max, y_pred_range, y_model_r2
  S = sample characteristics: n, y_mean, x_var

Outputs: results/raw/meta_group_ablation.json,
         results/tables/meta_group_ablation.csv,
         results/compare/meta_group_ablation.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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
GROUPS = {
    "E_effect": ["logit_ace", "lpm_x_coef", "unadj_ace"],
    "A_association_diagnostics": [
        "logit_lpm_diff", "adj_unadj_diff", "z_coef_mag", "z_x_coef_mag",
        "z_x_r2", "y_pred_min", "y_pred_max", "y_pred_range", "y_model_r2",
    ],
    "S_sample": ["n", "y_mean", "x_var"],
}


def prepare(df, feats):
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(df[feats].median())
    return X, df["abs_bias_pct"].values


def fit_eval(train, test, feats):
    Xtr, ytr = prepare(train, feats)
    Xte, yte = prepare(test, feats)
    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    return rf, Xte, yte, {
        "n_features": len(feats),
        "r2_test": float(r2_score(yte, pred)),
        "mae_test": float(mean_absolute_error(yte, pred)),
    }


def grouped_permutation(rf, X, y, cols, n_repeats=10):
    base = r2_score(y, rf.predict(X))
    rng = np.random.RandomState(SEED)
    drops = []
    for _ in range(n_repeats):
        Xp = X.copy()
        idx = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].to_numpy()[idx]
        drops.append(base - r2_score(y, rf.predict(Xp)))
    return float(np.mean(drops)), float(np.std(drops))


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "meta_training_data.parquet")
    train = df[df["model_type"].isin(["M1", "M5"])]
    test = df[df["model_type"].isin(["M6", "M7"])]

    rf, Xte, yte, full = fit_eval(train, test, FEATURES)
    rows = [{"analysis": "full", "group": "all", **full}]

    for name, cols in GROUPS.items():
        mean, sd = grouped_permutation(rf, Xte, yte, cols)
        rows.append({
            "analysis": "grouped_permutation",
            "group": name,
            "n_features": len(cols),
            "r2_drop_mean": mean,
            "r2_drop_sd": sd,
        })
        remaining = [f for f in FEATURES if f not in cols]
        _, _, _, res = fit_eval(train, test, remaining)
        rows.append({"analysis": "drop_group", "group": name, **res})

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "meta_group_ablation.csv", index=False)
    (RAW / "meta_group_ablation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    lines = [
        "# Grouped feature importance and group removal (reviewer M4)",
        "",
        f"- full model: {full['n_features']} features, held-out R2={full['r2_test']:.4f}, "
        f"MAE={full['mae_test']:.2f}",
        "",
        "## Grouped permutation importance (R2 drop when the whole group is permuted)",
        "",
        "| group | features | mean R2 drop | SD |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        if r["analysis"] == "grouped_permutation":
            lines.append(
                f"| {r['group']} | {r['n_features']} | {r['r2_drop_mean']:.4f} | "
                f"{r['r2_drop_sd']:.4f} |"
            )
    lines += ["", "## Group removal (refit without the group)", "",
              "| group | remaining features | R2 | MAE |", "|---|---:|---:|---:|"]
    for r in rows:
        if r["analysis"] == "drop_group":
            lines.append(
                f"| {r['group']} | {r['n_features']} | {r['r2_test']:.4f} | "
                f"{r['mae_test']:.2f} |"
            )
    lines += ["", "## Full model", "",
              f"- R2={full['r2_test']:.4f}, MAE={full['mae_test']:.2f}", ""]
    (OUT / "meta_group_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {TABLES / 'meta_group_ablation.csv'}")
    print(f"Saved {OUT / 'meta_group_ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
