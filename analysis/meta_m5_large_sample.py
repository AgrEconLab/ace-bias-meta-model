#!/usr/bin/env python3
"""Large-sample check of the M5 direct/total effect mapping (response M3)."""
from __future__ import annotations

import json
import os

from dgp import get_true_ace_case, get_true_direct_effect_case, simulate_model
from estimators import logit_ace_gcomp

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "results", "raw", "meta_m5_large_sample.json")
N = 5_000_000
SEED = 42


def main() -> None:
    data = simulate_model("M5", "a", n=N, seed=SEED)
    adjusted = logit_ace_gcomp(data, bootstrap=False)["ace"]
    unadjusted = logit_ace_gcomp(data, z_name=None, bootstrap=False)["ace"]
    total = get_true_ace_case("M5", "a", n_mc=500_000, seed=SEED)
    direct = get_true_direct_effect_case("M5", "a", n_mc=500_000, seed=SEED)
    payload = {
        "N": N,
        "seed": SEED,
        "adjusted": adjusted,
        "direct_effect": direct,
        "unadjusted": unadjusted,
        "total_effect": total,
        "adjusted_minus_direct": adjusted - direct,
        "unadjusted_minus_total": unadjusted - total,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"[OK] Saved {OUT}")


if __name__ == "__main__":
    main()
