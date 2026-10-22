"""Positivity / propensity-overlap diagnostics (binary-X configurations).

For the binary-exposure case (case d), fits P(X=1|Z) for M1 and M3 (the two
DAGs with a structural Z->X edge) across lambda_zx and N, and reports:
  - min/max fitted propensity;
  - fraction of observations with propensity outside [0.05, 0.95];
  - share of datasets with any such observation;
  - separation failures.

M2/M4/M5/M6/M7 have X exogenous (Bernoulli(0.5)) after the DGP correction, so
their propensity is constant; they serve as a reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from dgp import simulate_model

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
TABLES = ROOT / "../results/tables"
OUT = ROOT / "../results/compare"
SEED = 20240625
N_ITER = 20
LAMBDA_ZX = [0.10, 0.30, 0.50]
SAMPLE_SIZES = [250, 1000, 5000]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for model in ["M1", "M3"]:
        for lam in LAMBDA_ZX:
            params = {
                "lambda_zx": lam,
                "lambda_zy": 0.30 if model == "M1" else 0.0,
                "lambda_xy": 0.10,
                "lambda_xz": 0.30,
                "lambda_yz": 0.30,
                "intercept": 0.0,
            }
            for n in SAMPLE_SIZES:
                for it in range(N_ITER):
                    seed = SEED + it
                    data = simulate_model(model, "d", n=n, params=params, seed=seed)
                    try:
                        fit = sm.Logit.from_formula("X ~ Z", data=data).fit(
                            disp=False, maxiter=100
                        )
                        ps = fit.predict(data)
                        records.append({
                            "model_type": model,
                            "lambda_zx": lam,
                            "n": n,
                            "iter": it,
                            "ps_min": float(ps.min()),
                            "ps_max": float(ps.max()),
                            "violation_rate": float(((ps < 0.05) | (ps > 0.95)).mean()),
                            "separation": False,
                        })
                    except Exception as e:
                        records.append({
                            "model_type": model,
                            "lambda_zx": lam,
                            "n": n,
                            "iter": it,
                            "ps_min": np.nan,
                            "ps_max": np.nan,
                            "violation_rate": np.nan,
                            "separation": True,
                            "error": str(e)[:120],
                        })

    df = pd.DataFrame(records)
    df.to_csv(TABLES / "meta_positivity.csv", index=False)

    def summ(g):
        return {
            "n_datasets": int(len(g)),
            "separation_share": float(g["separation"].mean()),
            "mean_ps_min": float(g["ps_min"].mean()),
            "mean_ps_max": float(g["ps_max"].mean()),
            "mean_violation_rate": float(g["violation_rate"].mean()),
            "share_datasets_with_violation": float(
                (g["violation_rate"] > 0).mean()
            ),
        }

    by = []
    for (model, lam, n), g in df.groupby(["model_type", "lambda_zx", "n"]):
        s = summ(g)
        s.update({"model_type": model, "lambda_zx": lam, "n": n})
        by.append(s)
    by_df = pd.DataFrame(by)
    by_df.to_csv(TABLES / "meta_positivity_summary.csv", index=False)
    out = {
        "description": "propensity overlap for binary-X M1/M3",
        "by_model_lambda_n": by_df.to_dict("records"),
        "overall": summ(df),
    }
    (RAW / "meta_positivity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    lines = [
        "# Positivity / propensity-overlap diagnostics (binary X)",
        "",
        f"- datasets: {len(df)} (M1/M3, case d; {N_ITER} iterations per cell)",
        "",
        "| model | lambda_zx | N | mean min ps | mean max ps | mean violation rate | share with violation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by:
        lines.append(
            f"| {r['model_type']} | {r['lambda_zx']:.2f} | {r['n']} | "
            f"{r['mean_ps_min']:.3f} | {r['mean_ps_max']:.3f} | "
            f"{r['mean_violation_rate']:.3f} | {r['share_datasets_with_violation']:.1%} |"
        )
    lines += [
        "",
        "Note: M2/M4/M5/M6/M7 have exogenous X (Bernoulli(0.5)) after the DGP",
        "correction; their propensity is constant and positivity is not a concern.",
        "",
    ]
    (OUT / "meta_positivity.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {TABLES / 'meta_positivity.csv'}")
    print(f"Saved {OUT / 'meta_positivity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
