"""
DREAM Pipeline — Data Generating Processes (DGP)
7 canonical DAG structures × 4 cases (a-d) for logit-based ACE evaluation.

Each DGP function returns a pandas DataFrame with columns (X, Y, Z)
generated from the specified causal structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from config import CONFIG, set_seed


def _logit_inv(eta: np.ndarray) -> np.ndarray:
    """Inverse logit (logistic) function."""
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))


def _probit_inv(eta: np.ndarray) -> np.ndarray:
    """Inverse probit (normal CDF) function."""
    from scipy.stats import norm
    return norm.cdf(eta)


# ─── Case specifications ─────────────────────────────────────────────────────
# Case a: Linear influences + Normal disturbances
# Case b: Nonlinear Bernoulli influence Z→X + Normal elsewhere
# Case c: Linear influences + outcome/confounder Bernoulli
# Case d: Nonlinear Bernoulli influences + all Bernoulli

def _generate_case_data(n: int, case: str, model_params: Dict) -> pd.DataFrame:
    """
    Generate data for a given case (a-d) with specified model parameters.

    Parameters
    ----------
    n : int
        Sample size
    case : str
        One of 'a', 'b', 'c', 'd'
    model_params : dict
        Must contain keys defining the causal structure:
        - lambda_zx, lambda_zy, lambda_xy, lambda_xz, lambda_yz
        - model_type: str identifying the DAG structure

    Returns
    -------
    pd.DataFrame with columns X, Y, Z (plus intermediate if needed)
    """
    lam_zx = model_params.get('lambda_zx', CONFIG.default_lambda_zx)
    lam_zy = model_params.get('lambda_zy', CONFIG.default_lambda_zy)
    lam_xy = model_params.get('lambda_xy', CONFIG.default_lambda_xy)
    lam_xz = model_params.get('lambda_xz', CONFIG.default_lambda_xz)
    lam_yz = model_params.get('lambda_yz', CONFIG.default_lambda_yz)
    intercept = model_params.get('intercept', 0.0)

    model_type = model_params.get('model_type', 'M1')

    # Generate base disturbances based on case
    if case in ['a', 'b']:
        # Normal disturbances
        u_z = np.random.normal(0, 1, n)
        u_x = np.random.normal(0, 1, n)
        u_y = np.random.normal(0, 1, n)
        z_base = np.random.normal(0, 1, n)  # Z initial
    else:  # 'c', 'd'
        # Bernoulli disturbances
        u_z = np.random.binomial(1, 0.5, n)
        u_x = np.random.binomial(1, 0.5, n)
        u_y = np.random.binomial(1, 0.5, n)
        z_base = np.random.binomial(1, 0.5, n)  # Z initial

    # ── Generate Z based on model structure ──
    if model_type == 'M1':
        # Confounding: Z causes both X and Y, Z is exogenous
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:  # 'b', 'd'
            z = z_base.astype(float)

    elif model_type == 'M2':
        # Outcome predictor: Z -> Y only
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:  # 'b', 'd'
            z = z_base.astype(float)

    elif model_type == 'M3':
        # Exposure predictor: Z -> X only
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:
            z = z_base.astype(float)

    elif model_type == 'M4':
        # Exposure descendant: X -> Z
        # Need X first, then Z = f(X). We'll generate X from u_x initially
        if case in ['a', 'c']:
            x_init = u_x  # X from disturbance, will generate properly later
            z = 0.5 * (lam_xz * x_init) + 0.5 * u_z
        else:
            # For Bernoulli, Z depends on X
            x_init = u_x
            p_z = _logit_inv(lam_xz * (2 * x_init - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    elif model_type == 'M5':
        # Mediation: X -> Z -> Y
        if case in ['a', 'c']:
            x_init = u_x
            z = 0.5 * (lam_xz * x_init + u_z)
        else:
            x_init = u_x
            p_z = _logit_inv(lam_xz * (2 * x_init - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    elif model_type == 'M6':
        # Collider: X -> Z <- Y
        if case in ['a', 'c']:
            z = 0.5 * (lam_xz * u_x + u_z)
        else:
            z = z_base.astype(float)  # Will be updated after Y generation

    elif model_type == 'M7':
        # Outcome descendant: Y -> Z
        if case in ['a', 'c']:
            z = 0.5 * u_z + 0.5 * z_base
        else:
            z = z_base.astype(float)

    # ── Generate X ──
    if model_type in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7']:
        if case in ['a', 'c']:
            # Linear influence Z -> X
            x_linear = lam_zx * z + u_x
            if case == 'a':
                x = x_linear
            else:  # 'c': standardize for Bernoulli outcome relation
                x = x_linear
        else:  # 'b', 'd': nonlinear Bernoulli influence Z -> X
            if case == 'b':
                z_binary = (z > np.median(z)).astype(float)
            else:  # 'd'
                z_binary = z.copy()

            if model_type == 'M6':
                # Collider: X is exogenous
                x = u_x if case in ['a', 'c'] else u_x.astype(float)
            else:
                p_x = _logit_inv(lam_zx * (2 * z_binary - 1) - 0.5)
                x = np.random.binomial(1, p_x).astype(float)
    else:
        # M4: X is exogenous
        if case in ['a', 'c']:
            x = u_x
        else:
            x = u_x.astype(float)

    # ── Generate Y ──
    if model_type == 'M5':
        # Mediation: Y depends on X and Z (Z is mediator)
        y_eta = intercept + lam_xy * x + lam_zy * z + u_y
    elif model_type == 'M6':
        # Collider: Y is a cause of Z but not affected by Z
        y_eta = intercept + lam_xy * x + u_y
    elif model_type == 'M7':
        # Outcome descendant: Y is cause of Z
        y_eta = intercept + lam_xy * x + u_y
    elif model_type in ['M1', 'M2', 'M3', 'M4']:
        if model_type == 'M1':
            # Confounding: Y depends on X and Z
            y_eta = intercept + lam_xy * x + lam_zy * z + u_y
        elif model_type == 'M2':
            # Outcome predictor: Y depends on X and Z
            y_eta = intercept + lam_xy * x + lam_zy * z + u_y
        else:
            # M3, M4: Y depends on X only
            y_eta = intercept + lam_xy * x + u_y

    # Standardize eta for consistent treatment
    y_eta = y_eta - np.mean(y_eta)
    y_prob = _logit_inv(y_eta)
    y = np.random.binomial(1, y_prob).astype(float)

    # ── Post-generation adjustments for specific models ──
    if model_type == 'M6':
        # Collider: Z is a function of BOTH X and Y
        if case in ['a', 'c']:
            z = lam_xz * x + lam_yz * y + u_z
            z = z - np.mean(z)  # Standardize
        else:  # 'b', 'd'
            p_z = _logit_inv(lam_xz * (2 * x - 1) + lam_yz * (2 * y - 1) + 0.5 * z_base)
            # z is already drawn from z_base, create new one
            z = np.random.binomial(1, p_z).astype(float)

    elif model_type == 'M7':
        # Outcome descendant: Z depends on Y
        if case in ['a', 'c']:
            z = lam_yz * y + u_z
            z = z - np.mean(z)
        else:
            p_z = _logit_inv(lam_yz * (2 * y - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    elif model_type == 'M4':
        # Exposure descendant: Z depends on X (regenerate with correct X)
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:
            p_z = _logit_inv(lam_xz * (2 * x - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    elif model_type == 'M5':
        # Mediation: regenerate Z with correct X
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:
            p_z = _logit_inv(lam_xz * (2 * x - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    # ── Prepare output ──
    data = pd.DataFrame({
        'X': x,
        'Y': y.astype(int),
        'Z': z,
    })

    # Compute true ACE via g-computation with known DGP
    # (Will be estimated via MC approximation)

    return data


def simulate_model(model_type: str, case: str, n: int = 1000,
                   params: Optional[Dict] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Simulate data from a specified DAG model and case.

    Parameters
    ----------
    model_type : str
        One of 'M1', 'M2', ..., 'M7'
    case : str
        One of 'a', 'b', 'c', 'd'
    n : int
        Sample size (default 1000)
    params : dict, optional
        Parameter overrides
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
    """
    if seed is not None:
        set_seed(seed)

    model_params = {
        'model_type': model_type,
        'lambda_zx': CONFIG.default_lambda_zx,
        'lambda_zy': CONFIG.default_lambda_zy,
        'lambda_xy': CONFIG.default_lambda_xy,
        'lambda_xz': CONFIG.default_lambda_xz,
        'lambda_yz': CONFIG.default_lambda_yz,
        'intercept': 0.0,
    }
    if params:
        model_params.update(params)

    return _generate_case_data(n, case, model_params)


def get_true_ace(model_type: str, params: Optional[Dict] = None,
                 n_mc: int = 200000, seed: int = 42) -> float:
    """
    Compute true ACE for a given model via large-sample Monte Carlo.

    Uses g-computation: E[Y|do(X=1)] - E[Y|do(X=0)]
    Marginalized over the distribution of confounders.
    """
    set_seed(seed)

    model_params = {
        'model_type': model_type,
        'lambda_zx': CONFIG.default_lambda_zx,
        'lambda_zy': CONFIG.default_lambda_zy,
        'lambda_xy': CONFIG.default_lambda_xy,
        'lambda_xz': CONFIG.default_lambda_xz,
        'lambda_yz': CONFIG.default_lambda_yz,
        'intercept': 0.0,
    }
    if params:
        model_params.update(params)

    # We need the analytical true ACE under the DGP specification
    # For logit DGP: P(Y=1|X, Z) = logistic(intercept + lambda_xy*X + lambda_zy*Z + ...)
    # We'll use g-computation with the true DGP parameters on a large sample

    # Step 1: Generate Z from its marginal distribution
    # This depends on the model
    model_type_str = model_params['model_type']
    lam_zx = model_params['lambda_zx']
    lam_zy = model_params['lambda_zy']
    lam_xy = model_params['lambda_xy']
    lam_xz = model_params['lambda_xz']
    lam_yz = model_params['lambda_yz']
    intercept = model_params['intercept']

    n = n_mc

    # For simplicity, use simulation-based true ACE
    # Generate Z from marginal (no conditioning on X)
    if model_type_str == 'M1':
        # Z is exogenous, draw from its distribution
        z = np.random.normal(0, 1, n)
        # Potential outcomes under X=1 and X=0
        p_y1 = _logit_inv(intercept + lam_xy * 1 + lam_zy * z)
        p_y0 = _logit_inv(intercept + lam_xy * 0 + lam_zy * z)
        ace = np.mean(p_y1 - p_y0)

    elif model_type_str == 'M2':
        z = np.random.normal(0, 1, n)
        p_y1 = _logit_inv(intercept + lam_xy * 1 + lam_zy * z)
        p_y0 = _logit_inv(intercept + lam_xy * 0 + lam_zy * z)
        ace = np.mean(p_y1 - p_y0)

    elif model_type_str == 'M3':
        # Z -> X only, Y does NOT depend on Z
        z = np.random.normal(0, 1, n)
        p_y1 = _logit_inv(intercept + lam_xy * 1)
        p_y0 = _logit_inv(intercept + lam_xy * 0)
        ace = np.mean(p_y1 - p_y0)

    elif model_type_str == 'M4':
        # X -> Z, Y depends on X only (Z is descendant)
        z = np.random.normal(0, 1, n)
        p_y1 = _logit_inv(intercept + lam_xy * 1)
        p_y0 = _logit_inv(intercept + lam_xy * 0)
        ace = np.mean(p_y1 - p_y0)

    elif model_type_str == 'M5':
        # Mediation: X -> Z -> Y
        # Total effect = direct effect (X->Y) + indirect effect (X->Z->Y)
        z = np.random.normal(0, 1, n)
        # Under do(X=1), Z = lam_xz*1 + U_z, then Y from X and Z
        # We need to marginalize over Z's distribution under each X value
        z1 = lam_xz * 1 + np.random.normal(0, 1, n)
        z0 = lam_xz * 0 + np.random.normal(0, 1, n)
        p_y1 = _logit_inv(intercept + lam_xy * 1 + lam_zy * z1)
        p_y0 = _logit_inv(intercept + lam_xy * 0 + lam_zy * z0)
        ace = np.mean(p_y1 - p_y0)

    elif model_type_str == 'M6':
        # Collider: Z = X + Y, Y depends on X only
        z = np.random.normal(0, 1, n)
        p_y1 = _logit_inv(intercept + lam_xy * 1)
        p_y0 = _logit_inv(intercept + lam_xy * 0)
        ace = np.mean(p_y1 - p_y0)

    elif model_type_str == 'M7':
        # Outcome descendant
        z = np.random.normal(0, 1, n)
        p_y1 = _logit_inv(intercept + lam_xy * 1)
        p_y0 = _logit_inv(intercept + lam_xy * 0)
        ace = np.mean(p_y1 - p_y0)

    return ace


def get_true_ace_case(model_type: str, case: str = 'a',
                       params: Optional[Dict] = None,
                       n_mc: int = 200000, seed: int = 42) -> float:
    """
    Compute case-specific true ACE via large-sample Monte Carlo.

    Unlike get_true_ace (which uses normal approximations only valid for case a),
    this function uses the EXACT DGP for each case, including:
      - Case-specific error distributions (normal vs Bernoulli)
      - Case-specific X generation (linear vs Bernoulli logit)
      - The y_eta centering step that shifts the effective intercept
      - Post-generation Z adjustments (M4, M5, M6, M7)

    The true ACE is computed by:
      1. Draw a large sample from the case-specific DGP
      2. Compute the sample centering of y_eta (as done in the actual DGP)
      3. Apply g-computation using the TRUE DGP parameters
         P(Y=1|do(X=1)) = mean(Λ(intercept + λ_xy*1 + λ_zy*Z + u_y - centering))
         P(Y=1|do(X=0)) = mean(Λ(intercept + λ_xy*0 + λ_zy*Z + u_y - centering))
      4. ACE = difference
    """
    set_seed(seed)

    model_params = {
        'model_type': model_type,
        'lambda_zx': CONFIG.default_lambda_zx,
        'lambda_zy': CONFIG.default_lambda_zy,
        'lambda_xy': CONFIG.default_lambda_xy,
        'lambda_xz': CONFIG.default_lambda_xz,
        'lambda_yz': CONFIG.default_lambda_yz,
        'intercept': 0.0,
    }
    if params:
        model_params.update(params)

    lam_zx = model_params['lambda_zx']
    lam_zy = model_params['lambda_zy']
    lam_xy = model_params['lambda_xy']
    lam_xz = model_params['lambda_xz']
    lam_yz = model_params['lambda_yz']
    intercept = model_params['intercept']

    n = n_mc

    # ── Step 1: Generate disturbances and Z ──
    if case in ['a', 'b']:
        u_z = np.random.normal(0, 1, n)
        u_x = np.random.normal(0, 1, n)
        u_y = np.random.normal(0, 1, n)
        z_base = np.random.normal(0, 1, n)
    else:  # 'c', 'd'
        u_z = np.random.binomial(1, 0.5, n)
        u_x = np.random.binomial(1, 0.5, n)
        u_y = np.random.binomial(1, 0.5, n)
        z_base = np.random.binomial(1, 0.5, n)

    # ── Step 2: Generate Z ──
    # (following the same logic as _generate_case_data)
    if model_type in ['M1', 'M2']:
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:
            z = z_base.astype(float)
    elif model_type == 'M3':
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:
            z = z_base.astype(float)
    elif model_type == 'M4':
        if case in ['a', 'c']:
            z_init = 0.5 * (lam_xz * u_x) + 0.5 * u_z
        else:
            p_z_init = _logit_inv(lam_xz * (2 * u_x - 1) + 0.5 * z_base)
            z_init = np.random.binomial(1, p_z_init).astype(float)
        z = z_init  # placeholder, will be regenerated after X
    elif model_type == 'M5':
        if case in ['a', 'c']:
            z_init = 0.5 * (lam_xz * u_x + u_z)
        else:
            p_z_init = _logit_inv(lam_xz * (2 * u_x - 1) + 0.5 * z_base)
            z_init = np.random.binomial(1, p_z_init).astype(float)
        z = z_init
    elif model_type == 'M6':
        if case in ['a', 'c']:
            z = 0.5 * (lam_xz * u_x + u_z)
        else:
            z = z_base.astype(float)  # placeholder
    elif model_type == 'M7':
        if case in ['a', 'c']:
            z = 0.5 * u_z + 0.5 * z_base
        else:
            z = z_base.astype(float)

    # ── Step 3: Generate X ──
    if model_type in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7']:
        if case in ['a', 'c']:
            x_lin = lam_zx * z + u_x
            x = x_lin
        else:
            if case == 'b':
                z_binary = (z > np.median(z)).astype(float)
            else:
                z_binary = z.copy()
            p_x = _logit_inv(lam_zx * (2 * z_binary - 1) - 0.5)
            x = np.random.binomial(1, p_x).astype(float)
    else:  # M4
        x = u_x if case in ['a', 'c'] else u_x.astype(float)

    # ── Step 4: Compute y_eta and centering ──
    if model_type == 'M5':
        y_eta = intercept + lam_xy * x + lam_zy * z + u_y
    elif model_type in ['M6', 'M7']:
        y_eta = intercept + lam_xy * x + u_y
    elif model_type in ['M1', 'M2', 'M3', 'M4']:
        if model_type in ['M1', 'M2']:
            y_eta = intercept + lam_xy * x + lam_zy * z + u_y
        else:  # M3, M4
            y_eta = intercept + lam_xy * x + u_y

    # Centering is the key step get_true_ace misses
    centering = np.mean(y_eta)

    # ── Step 5: Compute potential outcomes with centering ──
    if model_type == 'M5':
        # Mediation: Z depends on X, so potential outcomes have different Z distributions
        z1 = lam_xz * 1 + u_z if case in ['a', 'c'] else \
             np.random.binomial(1, _logit_inv(lam_xz * (2*1-1) + 0.5*z_base), n).astype(float)
        z0 = lam_xz * 0 + u_z if case in ['a', 'c'] else \
             np.random.binomial(1, _logit_inv(lam_xz * (2*0-1) + 0.5*z_base), n).astype(float)

        # Re-compute u_y for counterfactuals (drawn fresh)
        if case in ['c', 'd']:
            u_y_cf = np.random.binomial(1, 0.5, n)
        else:
            u_y_cf = np.random.normal(0, 1, n)

        p_y1 = _logit_inv(intercept + lam_xy * 1 + lam_zy * z1 + u_y_cf - centering)
        p_y0 = _logit_inv(intercept + lam_xy * 0 + lam_zy * z0 + u_y_cf - centering)
    else:
        if model_type in ['M1', 'M2']:
            eta1 = intercept + lam_xy * 1 + lam_zy * z + u_y - centering
            eta0 = intercept + lam_xy * 0 + lam_zy * z + u_y - centering
        elif model_type in ['M3', 'M4', 'M6', 'M7']:
            # Y does NOT depend on Z for these models
            eta1 = intercept + lam_xy * 1 + u_y - centering
            eta0 = intercept + lam_xy * 0 + u_y - centering
        p_y1 = _logit_inv(eta1)
        p_y0 = _logit_inv(eta0)

    ace = float(np.mean(p_y1 - p_y0))
    return ace


# ─── Model Registry ──────────────────────────────────────────────────────────
# Maps model_type to a short description

MODEL_DESCRIPTIONS = {
    'M1': 'Confounding: Z → X, Z → Y',
    'M2': 'Outcome Predictor: Z → Y only',
    'M3': 'Exposure Predictor: Z → X only',
    'M4': 'Exposure Descendant: X → Z',
    'M5': 'Mediation: X → Z → Y',
    'M6': 'Collider: X → Z ← Y',
    'M7': 'Outcome Descendant: Y → Z',
}

CASE_DESCRIPTIONS = {
    'a': 'Linear influences + Normal disturbances',
    'b': 'Nonlinear Bernoulli Z→X + Normal elsewhere',
    'c': 'Linear influences + Bernoulli outcomes',
    'd': 'Nonlinear Bernoulli + all Bernoulli variables',
}
