"""Negative-effect check for the fixed risk-zone rule.

Generates a targeted negative-effect grid (lambda_xy < 0), applies the fixed
two-input rule estimated on the positive-effect training data, and reports the
error distribution by zone. This tests whether a negative LPM coefficient still
means "high error" when the true effect is protective.

Targeted grid: 7 DAGs x 2 cases x 3 negative lambda x 1 path strength x 3 N,
20 iterations = 126 configs x 20 = 2,520 datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import meta_generate as mg
from dgp import simulate_model, get_true_ace_case
from estimators import logit_ace_gcomp

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
TABLES = ROOT / "../results/tables"
OUT = ROOT / "../results/compare"
SEED = 20240625
N_ITER = 20

# Targeted negative-effect grid
mg.LAMBDA_XY = [-0.05, -0.10, -0.30]
mg.LAMBDA_PATH = [0.30]
mg.SAMPLE_SIZES = [250, 1000, 5000]
mg.CASES = ["a", "d"]

NEG_LAMBDA = [-0.05, -0.10, -0.30]


def zone_for(lpm_x, n):
    if lpm_x <= -0.024:
        return "RED"
    if -0.002 < lpm_x <= 0.102:
        return "GREEN"
    if -0.024 < lpm_x <= -0.002 or (lpm_x > 0.102 and n > 625):
        return "YELLOW-A"
    if lpm_x > 0.102 and n <= 625:
        return "YELLOW-B"
    return "YELLOW-A"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    configs = list(mg._generate_configs())
    print(f"negative-effect configs: {len(configs)}")

    records = []
    for idx, (model, case, n, params) in enumerate(configs):
        true_ace = get_true_ace_case(model, case, params, n_mc=50000, seed=42)
        if (idx + 1) % 25 == 0:
            print(f"  config {idx+1}/{len(configs)}")
        for it in range(N_ITER):
            seed = SEED + idx + it * 10000
            data = simulate_model(model, case, n=n, params=params, seed=seed)
            logit_ace = logit_ace_gcomp(data, bootstrap=False)["ace"]
            lpm = sm.OLS.from_formula("Y ~ X + Z", data=data).fit()
            lpm_x = float(lpm.params.get("X", 0))
            abs_err = abs(logit_ace - true_ace)
            records.append({
                "model_type": model,
                "case_type": case,
                "n": n,
                "lambda_xy": params["lambda_xy"],
                "true_ace": true_ace,
                "logit_ace": logit_ace,
                "lpm_x_coef": lpm_x,
                "abs_error_pp": abs_err,
                "abs_bias_pct": abs_err / abs(true_ace) * 100 if true_ace != 0 else np.nan,
                "zone": zone_for(lpm_x, n),
            })

    df = pd.DataFrame(records)
    df.to_csv(TABLES / "meta_negative_effect.csv", index=False)

    def summarize(g):
        return {
            "n": int(len(g)),
            "median_abs_error_pp": float(g["abs_error_pp"].median()),
            "mean_abs_error_pp": float(g["abs_error_pp"].mean()),
            "median_abs_rel_error_pct": float(g["abs_bias_pct"].median()),
            "high_error_rate": float((g["abs_bias_pct"] > 5).mean()),
            "mean_lpm_x_coef": float(g["lpm_x_coef"].mean()),
        }

    overall = summarize(df)
    by_zone = {z: summarize(g) for z, g in df.groupby("zone")}
    by_dag = {m: summarize(g) for m, g in df.groupby("model_type")}
    out = {
        "description": "targeted negative-effect grid; fixed positive-effect rule",
        "n_rows": len(df),
        "overall": overall,
        "by_zone": by_zone,
        "by_dag": by_dag,
    }
    (RAW / "meta_negative_effect.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    lines = [
        "# Negative-effect check of the fixed risk-zone rule",
        "",
        f"- targeted grid: {len(configs)} configs x {N_ITER} = {len(df)} datasets",
        "- lambda_xy in {-0.05, -0.10, -0.30}; one path strength (0.30)",
        "",
        "## Overall",
        "",
        f"- median |error|: {overall['median_abs_error_pp']:.4f} pp",
        f"- median relative error: {overall['median_abs_rel_error_pct']:.1f}%",
        f"- high-error rate (>5% relative): {overall['high_error_rate']:.1%}",
        "",
        "## By fixed-rule zone",
        "",
        "| zone | n | median |error| (pp) | median rel. error | high-error rate | mean lpm_x_coef |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for z in ["RED", "GREEN", "YELLOW-A", "YELLOW-B"]:
        if z not in by_zone:
            continue
        s = by_zone[z]
        lines.append(
            f"| {z} | {s['n']} | {s['median_abs_error_pp']:.4f} | "
            f"{s['median_abs_rel_error_pct']:.1f}% | {s['high_error_rate']:.1%} | "
            f"{s['mean_lpm_x_coef']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "If the RED zone contains many protective effects with low absolute error,",
        "the rule is not sign-symmetric and should be restricted to positive effects.",
        "",
    ]
    (OUT / "meta_negative_effect.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {TABLES / 'meta_negative_effect.csv'}")
    print(f"Saved {RAW / 'meta_negative_effect.json'}")
    print(f"Saved {OUT / 'meta_negative_effect.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
