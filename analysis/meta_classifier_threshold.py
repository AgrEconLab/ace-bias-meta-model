"""High-error classification at alternative AREE thresholds.

The 5% high-error indicator is almost always 1 (93.9% in the training set), so
the classifier's 0.955 accuracy and Cohen's kappa of 0.000 reproduce the
always-high rule. This script asks whether the negative result is specific to
the threshold: it repeats the logistic-regression classifier at 5%, 25%, and
50% AREE on the primary held-out split (M1/M5 -> M6/M7) and on the stratified
random 80/20 split.

Outputs:
  results/raw/meta_classifier_threshold.json
  results/tables/meta_classifier_threshold.csv
  results/compare/meta_classifier_threshold.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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
THRESHOLDS = [5, 25, 50]


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    x_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians)
    x_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians)
    return x_train, x_test


def evaluate(train: pd.DataFrame, test: pd.DataFrame, threshold: int) -> dict:
    x_train, x_test = prepare(train, test)
    y_train = (train["abs_bias_pct"] > threshold).astype(int).to_numpy()
    y_test = (test["abs_bias_pct"] > threshold).astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return {
            "threshold": threshold,
            "error": "single class in training set",
            "n_train_positive": int(y_train.sum()),
            "n_train": int(len(y_train)),
            "n_test_positive": int(y_test.sum()),
            "n_test": int(len(y_test)),
        }
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n_train": int(len(y_train)),
        "n_train_positive": int(y_train.sum()),
        "positive_share_train": float(y_train.mean()),
        "n_test": int(len(y_test)),
        "n_test_positive": int(y_test.sum()),
        "positive_share_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "meta_training_data.parquet")

    primary_train = df[df["model_type"].isin(["M1", "M5"])]
    primary_test = df[df["model_type"].isin(["M6", "M7"])]
    random_train, random_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["model_type"]
    )

    payload = {"meta": {"seed": SEED, "features": FEATURES,
                        "thresholds": THRESHOLDS}, "regimes": {}}
    rows = []
    for regime, train, test in [
        ("primary_heldout", primary_train, primary_test),
        ("random_stratified", random_train, random_test),
    ]:
        results = [evaluate(train, test, threshold) for threshold in THRESHOLDS]
        payload["regimes"][regime] = {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "results": results,
        }
        for result in results:
            rows.append({"regime": regime, **result})
        print(f"\n=== {regime} ===")
        for result in results:
            print(f"  threshold {result['threshold']:>2}%: "
                  f"pos_train={result.get('positive_share_train', float('nan')):.3f} "
                  f"pos_test={result.get('positive_share_test', float('nan')):.3f} "
                  f"kappa={result.get('kappa', float('nan')):.3f} "
                  f"balanced_acc={result.get('balanced_accuracy', float('nan')):.3f} "
                  f"auc={result.get('auc', float('nan')):.3f}")

    (RAW / "meta_classifier_threshold.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(
        TABLES / "meta_classifier_threshold.csv", index=False
    )

    lines = [
        "# High-error classification at alternative thresholds",
        "",
        "Logistic regression on the same 15 features, with the binary target "
        "`AREE > threshold`. The 5% threshold is the manuscript's original "
        "high-error indicator.",
        "",
        "| regime | threshold | positive share (train) | positive share (test) | "
        "accuracy | precision | recall | kappa | balanced accuracy | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['regime']} | {row['threshold']}% | "
                         f"{row['n_train_positive']}/{row['n_train']} | "
                         f"{row['n_test_positive']}/{row['n_test']} | "
                         f"n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {row['regime']} | {row['threshold']}% | "
            f"{row['positive_share_train']:.3f} | "
            f"{row['positive_share_test']:.3f} | {row['accuracy']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['kappa']:.3f} | {row['balanced_accuracy']:.3f} | "
            f"{row['auc']:.3f} |"
        )
    lines += ["", "Confusion matrices are in "
                  "`meta_classifier_threshold.json`.", ""]
    (OUT / "meta_classifier_threshold.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nSaved {RAW / 'meta_classifier_threshold.json'}")
    print(f"Saved {TABLES / 'meta_classifier_threshold.csv'}")
    print(f"Saved {OUT / 'meta_classifier_threshold.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
