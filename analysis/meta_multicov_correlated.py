"""Targeted correlated-covariate sensitivity (M8/M10).

Checks whether the max|Z coefficient| summary remains informative when Z1 and
Z2 are correlated. This is a targeted sensitivity, not the full multi-covariate
extension.

Grid: M8/M10 x rho in {0.0, 0.6} x n in {1000, 5000} x 20 iterations = 160
datasets, case a, default path strengths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "../results/raw"
TABLES = ROOT / "../results/tables"
OUT = ROOT / "../results/compare"
SEED = 20240625
N_ITER = 20
RHOS = [0.0, 0.6]
SIZES = [1000, 5000]


def simulate(model, n, rho, rs):
    u_x = rs.normal(0, 1, n)
    u_y = rs.normal(0, 1, n)
    e1 = rs.normal(0, 1, n)
    e2 = rs.normal(0, 1, n)
    z1 = e1
    z2 = rho * e1 + np.sqrt(1 - rho ** 2) * e2
    lam_xy, lam_zy, lam_zx = 0.10, 0.30, 0.30
    if model == "M8":
        x = lam_zx * z1 + u_x
        y_eta = lam_xy * x + lam_zy * z1 + u_y
    else:  # M10
        x = lam_zx * z1 + lam_zx * z2 + u_x
        y_eta = lam_xy * x + lam_zy * z1 + lam_zy * z2 + u_y
    y_eta = y_eta - np.mean(y_eta)
    y = rs.binomial(1, 1 / (1 + np.exp(-y_eta))).astype(float)
    return pd.DataFrame({"Y": y, "X": x, "Z1": z1, "Z2": z2})


def true_ace(model, rho, n_mc=100000, seed=42):
    rs = np.random.RandomState(seed)
    u_x = rs.normal(0, 1, n_mc)
    u_y = rs.normal(0, 1, n_mc)
    e1 = rs.normal(0, 1, n_mc)
    e2 = rs.normal(0, 1, n_mc)
    z1 = e1
    z2 = rho * e1 + np.sqrt(1 - rho ** 2) * e2
    lam_xy, lam_zy, lam_zx = 0.10, 0.30, 0.30
    if model == "M8":
        x = lam_zx * z1 + u_x
        eta_obs = lam_xy * x + lam_zy * z1 + u_y
        centering = np.mean(eta_obs)
        eta1 = lam_xy * 1 + lam_zy * z1 + u_y - centering
        eta0 = lam_xy * 0 + lam_zy * z1 + u_y - centering
    else:
        x = lam_zx * z1 + lam_zx * z2 + u_x
        eta_obs = lam_xy * x + lam_zy * z1 + lam_zy * z2 + u_y
        centering = np.mean(eta_obs)
        eta1 = lam_xy * 1 + lam_zy * z1 + lam_zy * z2 + u_y - centering
        eta0 = lam_xy * 0 + lam_zy * z1 + lam_zy * z2 + u_y - centering
    return float(np.mean(1 / (1 + np.exp(-eta1)) - 1 / (1 + np.exp(-eta0))))


def features(d):
    m = sm.Logit.from_formula("Y ~ X + Z1 + Z2", data=d).fit(disp=False, maxiter=100)
    coefs = [abs(m.params.get("Z1", 0)), abs(m.params.get("Z2", 0))]
    ols_x = sm.OLS.from_formula("X ~ Z1 + Z2", data=d).fit()
    ols_y = sm.OLS.from_formula("Y ~ X + Z1 + Z2", data=d).fit()
    lpm = sm.OLS.from_formula("Y ~ X + Z1 + Z2", data=d).fit(cov_type="HC3")
    pred = m.predict(d)
    return {
        "logit_ace": float(np.mean(pred[d["X"] == 1]) - np.mean(pred[d["X"] == 0]))
        if (d["X"] == 1).any() and (d["X"] == 0).any() else float("nan"),
        "lpm_x_coef": float(lpm.params.get("X", 0)),
        "z_coef_mag": float(max(coefs)),
        "z_x_coef_mag": float(max(abs(ols_x.params.get("Z1", 0)), abs(ols_x.params.get("Z2", 0)))),
        "z_x_r2": float(ols_x.rsquared),
        "y_model_r2": float(ols_y.rsquared),
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ["M8", "M10"]:
        for rho in RHOS:
            tace = true_ace(model, rho)
            for n in SIZES:
                rs = np.random.RandomState(SEED)
                for it in range(N_ITER):
                    d = simulate(model, n, rho, rs)
                    f = features(d)
                    # Adjusted g-computation ACE (logit)
                    m = sm.Logit.from_formula("Y ~ X + Z1 + Z2", data=d).fit(disp=False, maxiter=100)
                    d1 = d.copy(); d1["X"] = 1.0
                    d0 = d.copy(); d0["X"] = 0.0
                    ace = float(np.mean(m.predict(d1) - m.predict(d0)))
                    rows.append({
                        "model_type": model, "rho": rho, "n": n, "iter": it,
                        "true_ace": tace, "logit_ace": ace,
                        "abs_error_pp": abs(ace - tace),
                        **f,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "meta_multicov_correlated.csv", index=False)

    summary = {}
    for (model, rho), g in df.groupby(["model_type", "rho"]):
        feats = ["logit_ace", "lpm_x_coef", "z_coef_mag", "z_x_coef_mag", "z_x_r2", "y_model_r2"]
        X = g[feats].replace([np.inf, -np.inf], np.nan).fillna(g[feats].median()).to_numpy()
        y = g["abs_error_pp"].to_numpy()
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        pred = np.zeros_like(y)
        for tr, te in kf.split(X):
            rf = RandomForestRegressor(n_estimators=500, max_depth=6,
                                       min_samples_leaf=5, n_jobs=-1, random_state=SEED)
            rf.fit(X[tr], y[tr])
            pred[te] = rf.predict(X[te])
        summary[f"{model}_rho{rho}"] = {
            "n": int(len(g)),
            "mean_abs_error_pp": float(g["abs_error_pp"].mean()),
            "mean_z_coef_mag": float(g["z_coef_mag"].mean()),
            "corr_zcoef_error": float(g["z_coef_mag"].corr(g["abs_error_pp"])),
            "cv_r2": float(r2_score(y, pred)),
            "cv_mae": float(mean_absolute_error(y, pred)),
            "features": feats,
        }
    (RAW / "meta_multicov_correlated.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Correlated-covariate sensitivity (M8/M10)",
        "",
        "| model | rho | n | mean abs error (pp) | mean max|Z coef| | corr(max|Z|, error) | 5-fold CV R2 | CV MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary.items():
        model, rho = key.split("_rho")
        lines.append(
            f"| {model} | {rho} | {s['n']} | {s['mean_abs_error_pp']:.4f} | "
            f"{s['mean_z_coef_mag']:.3f} | {s['corr_zcoef_error']:.3f} | "
            f"{s['cv_r2']:.3f} | {s['cv_mae']:.4f} |"
        )
    lines += ["", "Targeted sensitivity only; not the full multi-covariate extension.", ""]
    (OUT / "meta_multicov_correlated.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {TABLES / 'meta_multicov_correlated.csv'}")
    print(f"Saved {OUT / 'meta_multicov_correlated.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
