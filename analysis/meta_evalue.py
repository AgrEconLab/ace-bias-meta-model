"""
E-value calibration: bridge meta-model bias predictions with sensitivity analysis.

Formulas follow VanderWeele & Ding (2017), Annals of Internal Medicine.
For a risk ratio RR > 1, the E-value is:
  E-value = RR + sqrt(RR * (RR - 1))

The calibration uses the meta-model's predicted absolute % bias b̂ to adjust
the observed ACE (probability-difference scale) before computing the E-value.

Reference:
  VanderWeele, T.J. and Ding, P. (2017). Sensitivity analysis in observational
  research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
"""

import numpy as np
import pandas as pd
import os
import sys

# ── paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(ROOT, "results", "tables")


def evalue_from_RR(RR):
    """Compute standard E-value for a risk ratio > 1."""
    RR = np.asarray(RR, dtype=float)
    assert np.all(RR > 0), "RR must be positive"
    out = np.where(RR <= 1, 1.0, RR + np.sqrt(RR * (RR - 1)))
    return out.item() if out.ndim == 0 else out


def calibrated_ace(ace_obs, bias_pct):
    """Bias-corrected ACE (most conservative, i.e., closest to zero).

    Parameters
    ----------
    ace_obs : float
        Observed ACE on the probability-difference scale.
    bias_pct : float
        Meta-model predicted absolute % bias (e.g., 62 for 62 pp).

    Returns
    -------
    ace_adj : float
        Bias-corrected ACE = ace_obs / (1 + bias_pct / 100).
        Always same sign as ace_obs, strictly closer to zero.
    """
    ace_obs = float(ace_obs)
    b = bias_pct / 100.0
    sign = np.sign(ace_obs) if ace_obs != 0 else 1.0
    return ace_obs / (1.0 + b)


def calibrated_evalue(ace_obs, prevalence, bias_pct):
    """Compute calibrated E-value given meta-model bias prediction.

    Parameters
    ----------
    ace_obs : float
        Observed ACE on the probability-difference scale.
    prevalence : float
        Outcome prevalence (y_mean), used to convert ACE to RR scale.
    bias_pct : float
        Meta-model predicted absolute % bias (pp).

    Returns
    -------
    dict with keys: ace_obs, ace_adj, RR_raw, RR_adj, evalue_raw, evalue_adj
    """
    ace_adj = calibrated_ace(ace_obs, bias_pct)
    # Convert ACE difference scale to risk ratio
    RR_raw = (prevalence + ace_obs) / prevalence if ace_obs > 0 else prevalence / (prevalence + ace_obs)
    RR_adj = (prevalence + ace_adj) / prevalence if ace_adj > 0 else prevalence / (prevalence + ace_adj)
    # Ensure RR >= 1 for E-value formula
    RR_raw_e = max(RR_raw, 1.0)
    RR_adj_e = max(RR_adj, 1.0)
    return {
        "ace_obs": ace_obs,
        "ace_adj": ace_adj,
        "RR_raw": RR_raw,
        "RR_adj": RR_adj,
        "evalue_raw": evalue_from_RR(RR_raw_e),
        "evalue_adj": evalue_from_RR(RR_adj_e),
    }


def zone_evalue_table(ace_obs=0.05, prevalence=0.40):
    """Build zone-specific calibration table using decision tree medians."""
    zones = {
        "GREEN":    {"bias_pct":  62, "condition": "lpm_x_coef in (-0.026, 0.102], logit_ace > -0.002"},
        "YELLOW-B": {"bias_pct": 221, "condition": "lpm_x_coef > -0.026, n <= 625"},
        "YELLOW-A": {"bias_pct": 179, "condition": "lpm_x_coef > 0.102, n > 625"},
        "RED":      {"bias_pct": 997, "condition": "lpm_x_coef <= -0.026"},
    }
    rows = []
    for zone, v in zones.items():
        res = calibrated_evalue(ace_obs, prevalence, v["bias_pct"])
        rows.append({
            "Zone": zone,
            "Condition": v["condition"],
            "Median |bias| (pp)": v["bias_pct"],
            "theta_adj / theta": f"{res['ace_adj'] / ace_obs:.2f}",
            "RR_raw": round(res["RR_raw"], 3),
            "RR_adj": round(res["RR_adj"], 3),
            "E-value (raw)": round(res["evalue_raw"], 2),
            "E-value (calibrated)": round(res["evalue_adj"], 2),
            "Attenuation": "Moderate" if zone == "GREEN" else ("Substantial" if "YELLOW" in zone else "Near-complete"),
        })
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("E-VALUE CALIBRATION: Meta-Model x Sensitivity Analysis")
    print("=" * 70)

    # 1. Verify against known example (VanderWeele & Ding 2017)
    print("\n--- Verification: known E-value examples ---")
    for RR in [1.3, 1.5, 2.0, 3.0]:
        ev = evalue_from_RR(RR)
        print(f"  RR = {RR:.1f}  ->  E-value = {ev:.3f}")

    # 2. Basic calibration example
    print("\n--- Calibration example ---")
    ace = 0.05       # observed ACE ~5pp
    prev = 0.40      # outcome prevalence 40%
    for bias_pp in [0, 62, 179, 997]:
        res = calibrated_evalue(ace, prev, bias_pp)
        print(f"  |bias|={bias_pp:3d}pp: theta_adj={res['ace_adj']:.4f}, "
              f"RR_raw={res['RR_raw']:.3f}->RR_adj={res['RR_adj']:.3f}, "
              f"E_raw={res['evalue_raw']:.2f}->E_adj={res['evalue_adj']:.2f}")

    # 3. Zone table
    print("\n--- Zone-specific calibration (ACE=0.05, prevalence=0.40) ---")
    df = zone_evalue_table(ace_obs=0.05, prevalence=0.40)
    print(df.to_string(index=False))

    # 4. Sensitivity: different ACE values
    print("\n--- Sensitivity to ACE magnitude (GREEN zone only) ---")
    for ace_test in [0.02, 0.05, 0.10, 0.20]:
        res = calibrated_evalue(ace_test, prevalence=0.40, bias_pct=62)
        print(f"  ACE={ace_test:.2f}: RR_raw={res['RR_raw']:.3f}, "
              f"E_raw={res['evalue_raw']:.2f} -> E_adj={res['evalue_adj']:.2f}")

    # 5. Save table
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, "evalue_calibration.csv"), index=False)
    print(f"\nTable saved to {TABLE_DIR}/evalue_calibration.csv")
    print("Done.")


if __name__ == "__main__":
    main()
