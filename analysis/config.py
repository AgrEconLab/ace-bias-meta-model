"""
DREAM Pipeline — Configuration & Global Parameters
Direction B: The Correction Frontier
"""

import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

for d in [RESULTS_DIR, RAW_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Seed Management ─────────────────────────────────────────────────────────
DEFAULT_SEED = 20240625


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    # Note: Python's random and hash seeds follow from numpy if not explicitly set
    import random
    random.seed(seed)


# ─── Simulation Parameters ───────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Master configuration for all simulation runs."""

    # Sample sizes
    sample_sizes: List[int] = field(default_factory=lambda: [250, 1000, 5000])

    # Monte Carlo iterations
    n_iterations: int = 1000

    # Default parameters (original paper values)
    default_lambda_zx: float = 0.3   # Z -> X
    default_lambda_zy: float = 0.3   # Z -> Y
    default_lambda_xy: float = 0.1   # X -> Y (causal effect)
    default_lambda_xz: float = 0.3   # X -> Z (mediation / collider)
    default_lambda_yz: float = 0.3   # Y -> Z (collider)

    # Sensitivity grid (dense at weak values to capture neutralization region)
    lambda_grid: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    # DAG models to simulate
    # M1=M_confounding, M2=M_outcome_predictor, M3=M_exposure_predictor,
    # M4=M_exposure_descendant, M5=M_mediation, M6=M_collider, M7=M_outcome_descendant
    dag_models: List[str] = field(default_factory=lambda: [f"M{i}" for i in range(1, 8)])

    # Cases (a=linear+normal, b=nonlinear_bernoulli+normal, c=linear+bernoulli, d=nonlinear+bernoulli_all)
    cases: List[str] = field(default_factory=lambda: ["a", "b", "c", "d"])

    # True ACE values (pre-computed via g-computation with large N)
    # These are approximations; exact values depend on parameters
    true_ace: Dict[str, float] = field(default_factory=lambda: {
        # Default params: lambda_zx=0.3, lambda_zy=0.3, lambda_xy=0.1
        "M1":  0.100,
        "M2":  0.100,
        "M3":  0.100,
        "M4":  0.100,
        "M5":  0.100,
        "M6":  0.100,
        "M7":  0.100,
    })

    # Bias threshold
    bias_threshold_pct: float = 5.0  # ±5%

    # Estimators to evaluate
    estimators: List[str] = field(default_factory=lambda: [
        "logit_ace_gcomp",      # Logit + g-computation (marginal standardization)
        "logit_ace_naive",      # Logit + naive marginal effect
        "probit_ace_gcomp",     # Probit + g-computation
        "tmle_ace",             # TMLE
        "ipw_ace",              # IPW
        "aipw_ace",             # Doubly robust AIPW
        "lpm_ace_ols",          # LPM / OLS + g-computation
    ])

    def __post_init__(self):
        # Force common random state
        set_seed(DEFAULT_SEED)


# Global singleton config
CONFIG = SimulationConfig()
