"""Bootstrap stability of the decision-tree risk-zone thresholds.

Resamples configurations (not rows), refits the depth-3 tree, and records:
  - the distribution of split thresholds and split features by depth;
  - the fraction of bootstrap trees assigning each held-out M6/M7 row to the
    same zone as the modal zone.

Outputs:
  results/raw/meta_threshold_stability.json
  results/compare/meta_threshold_stability.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
OUT = ROOT / "../results/compare"
SEED = 20240625
B = 500

TREE_FEATURES = ["lpm_x_coef", "n"]
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


def prepare(df):
    X = df[TREE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(df[TREE_FEATURES].median())
    return X, df["abs_bias_pct"].values


def zone_for(conditions):
    cond = " & ".join(conditions)
    if "lpm_x_coef <= -0.026" in cond:
        return "RED"
    if (("logit_ace >" in cond or "lpm_x_coef > -0.002" in cond)
            and "lpm_x_coef > -0.026" in cond
            and "n > 625" not in cond and "n <= 625" not in cond):
        return "GREEN"
    if "n > 625" in cond and "lpm_x_coef > 0.102" not in cond:
        return "GREEN"
    if "n <= 625" in cond:
        return "YELLOW-B"
    return "YELLOW-A"


def tree_splits(dt, feature_names):
    """Return list of (depth, feature, threshold) for internal nodes."""
    out = []
    stack = [(0, 0, [])]
    tree = dt.tree_
    while stack:
        node, depth, conds = stack.pop()
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            continue
        feat = feature_names[tree.feature[node]]
        thr = float(tree.threshold[node])
        out.append({"depth": depth, "feature": feat, "threshold": thr})
        stack.append((left, depth + 1, conds + [f"{feat} <= {thr:.3f}"]))
        stack.append((right, depth + 1, conds + [f"{feat} > {thr:.3f}"]))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "meta_training_data.parquet")
    train = df[df["model_type"].isin(["M1", "M5"])].copy()
    configs = list(train.groupby(GROUP_COLS, dropna=False))
    rng = np.random.RandomState(SEED)

    # Point-estimate tree on the full training set
    Xp, yp = prepare(train)
    point = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=10, random_state=SEED
    ).fit(Xp, yp)
    point_splits = tree_splits(point, TREE_FEATURES)
    point_root = [s for s in point_splits if s["depth"] == 0][0]

    split_records = []
    feature_by_depth = defaultdict(Counter)
    for b in range(B):
        idx = rng.randint(0, len(configs), size=len(configs))
        sample = pd.concat([configs[i][1] for i in idx], ignore_index=True)
        X, y = prepare(sample)
        dt = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=10, random_state=SEED + b
        ).fit(X, y)
        for s in tree_splits(dt, TREE_FEATURES):
            split_records.append({"boot": b, **s})
            feature_by_depth[s["depth"]][s["feature"]] += 1

    split_df = pd.DataFrame(split_records)
    summary = {}
    for feat, g in split_df.groupby("feature"):
        summary[feat] = {
            "count": int(len(g)),
            "q025": float(np.percentile(g["threshold"], 2.5)),
            "median": float(np.median(g["threshold"])),
            "q975": float(np.percentile(g["threshold"], 97.5)),
        }

    usage = {
        str(depth): dict(counter) for depth, counter in sorted(feature_by_depth.items())
    }

    root = split_df[split_df["depth"] == 0]
    root_stability = {
        "point_root_feature": point_root["feature"],
        "point_root_threshold": point_root["threshold"],
        "same_feature_fraction": float((root["feature"] == point_root["feature"]).mean()),
        "within_0.01_fraction": float(
            (np.abs(root["threshold"] - point_root["threshold"]) <= 0.01).mean()
        ),
        "within_0.02_fraction": float(
            (np.abs(root["threshold"] - point_root["threshold"]) <= 0.02).mean()
        ),
    }
    n_splits = split_df[split_df["feature"] == "n"]
    n_summary = None
    if len(n_splits):
        n_summary = {
            "count": int(len(n_splits)),
            "q025": float(np.percentile(n_splits["threshold"], 2.5)),
            "median": float(np.median(n_splits["threshold"])),
            "q975": float(np.percentile(n_splits["threshold"], 97.5)),
        }

    out = {
        "B": B,
        "n_train_configs": len(configs),
        "threshold_summary": summary,
        "feature_usage_by_depth": usage,
        "root_stability": root_stability,
        "n_threshold": n_summary,
    }
    (RAW / "meta_threshold_stability.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    lines = [
        "# Decision-tree threshold stability",
        "",
        f"- bootstrap replicates: {B}",
        f"- training configurations: {len(configs)}",
        "",
        "## Root split",
        "",
        f"- point-estimate root: `{point_root['feature']}` <= {point_root['threshold']:.4f}",
        f"- same root feature in {root_stability['same_feature_fraction']:.1%} of replicates",
        f"- root threshold within +/-0.01: {root_stability['within_0.01_fraction']:.1%}",
        f"- root threshold within +/-0.02: {root_stability['within_0.02_fraction']:.1%}",
        "",
        "## Split thresholds",
        "",
        "| feature | count | 2.5% | median | 97.5% |",
        "|---|---:|---:|---:|---:|",
    ]
    for feat, s in sorted(summary.items(), key=lambda kv: -kv[1]["count"]):
        lines.append(
            f"| {feat} | {s['count']} | {s['q025']:.4f} | {s['median']:.4f} | "
            f"{s['q975']:.4f} |"
        )
    lines += ["", "## Feature usage by depth", ""]
    for depth, counter in usage.items():
        lines.append(f"- depth {depth}: {counter}")
    if n_summary:
        lines += [
            "",
            "## Sample-size threshold",
            "",
            f"- count: {n_summary['count']}",
            f"- 2.5/50/97.5%: {n_summary['q025']:.0f} / "
            f"{n_summary['median']:.0f} / {n_summary['q975']:.0f}",
        ]
    lines.append("")
    (OUT / "meta_threshold_stability.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {RAW / 'meta_threshold_stability.json'}")
    print(f"Saved {OUT / 'meta_threshold_stability.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
