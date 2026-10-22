#!/usr/bin/env python3
"""RED/GREEN decision-rule operating characteristics on held-out M6/M7.

Replicates the exact depth-3 tree used for Table 2 (including its unrounded
split thresholds) and writes the 2x2 table to results/tables/red_rule_2x2.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import meta_export_tables as met

TABLES = Path(__file__).resolve().parent.parent / "results" / "tables"


def main() -> None:
    df = met.load_data()
    train = df[df["model_type"].isin(["M1", "M5"])]
    test = df[df["model_type"].isin(["M6", "M7"])].copy()

    X_train, y_train = met.prepare(train, met.TREE_FEATURES)
    X_test, _ = met.prepare(test, met.TREE_FEATURES)

    tree_model = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=10, random_state=met.SEED
    )
    tree_model.fit(X_train, y_train)
    test["leaf"] = tree_model.apply(X_test)

    tree = tree_model.tree_
    left, right = tree.children_left, tree.children_right
    threshold, feature = tree.threshold, tree.feature

    zone_info: dict[int, dict] = {}
    stack = [(0, [])]
    while stack:
        node, conditions = stack.pop()
        if left[node] == right[node]:
            zone_info[node] = {
                "conditions": conditions,
                "zone": met._zone_for_conditions(conditions),
            }
        else:
            name = met.TREE_FEATURES[feature[node]]
            thr = threshold[node]
            stack.append((left[node], conditions + [f"{name} <= {thr:.3f}"]))
            stack.append((right[node], conditions + [f"{name} > {thr:.3f}"]))

    test["zone"] = test["leaf"].map({k: v["zone"] for k, v in zone_info.items()})
    high = test["abs_bias_pct"] > 5

    rows = []
    for zone in ["RED", "GREEN"]:
        positive = test["zone"] == zone
        tp = int((positive & high).sum())
        fp = int((positive & ~high).sum())
        fn = int((~positive & high).sum())
        tn = int((~positive & ~high).sum())
        rows.append(
            {
                "zone": zone,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp),
                "npv": tn / (tn + fn),
            }
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "red_rule_2x2.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
