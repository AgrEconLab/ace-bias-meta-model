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

    # ── Z for M1/M2/M3 (exogenous) ──
    if model_type in ['M1', 'M2', 'M3']:
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:  # 'b', 'd'
            z = z_base.astype(float)
    # M4-M7: Z is generated after X (M4/M5) or after Y (M6/M7).

    # ── Generate X ──
    if model_type in ['M1', 'M3']:
        if case in ['a', 'c']:
            x = lam_zx * z + u_x
        else:  # 'b', 'd'
            if case == 'b':
                z_binary = (z > np.median(z)).astype(float)
            else:
                z_binary = z.copy()
            p_x = _logit_inv(lam_zx * (2 * z_binary - 1) - 0.5)
            x = np.random.binomial(1, p_x).astype(float)
    else:
        # M2, M4, M5, M6, M7: X is exogenous (no structural Z -> X edge)
        x = u_x.astype(float) if case in ['c', 'd'] else u_x

    # ── Generate Z for M4/M5 after X, before Y ──
    if model_type in ['M4', 'M5']:
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:  # 'b', 'd'
            p_z = _logit_inv(lam_xz * (2 * x - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    # ── Generate Y ──
    if model_type in ['M1', 'M2', 'M5']:
        y_eta = intercept + lam_xy * x + lam_zy * z + u_y
    else:
        # M3, M4, M6, M7: Y depends on X only
        y_eta = intercept + lam_xy * x + u_y

    # Standardize eta for consistent treatment
    y_eta = y_eta - np.mean(y_eta)
    y_prob = _logit_inv(y_eta)
    y = np.random.binomial(1, y_prob).astype(float)

    # ── Generate Z for M6/M7 after Y ──
    if model_type == 'M6':
        # Collider: Z is a function of BOTH X and Y
        if case in ['a', 'c']:
            z = lam_xz * x + lam_yz * y + u_z
            z = z - np.mean(z)
        else:  # 'b', 'd'
            p_z = _logit_inv(lam_xz * (2 * x - 1) + lam_yz * (2 * y - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    elif model_type == 'M7':
        # Outcome descendant: Z depends on Y
        if case in ['a', 'c']:
            z = lam_yz * y + u_z
            z = z - np.mean(z)
        else:  # 'b', 'd'
            p_z = _logit_inv(lam_yz * (2 * y - 1) + 0.5 * z_base)
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
    Compute the true TOTAL effect via Monte Carlo integration.

    Generation order matches the corrected _generate_case_data (2026-10-20):
      M1: Z exogenous -> X=f(Z,u_x); Y=f(X,Z,u_y)
      M2: Z exogenous, X exogenous; Y=f(X,Z,u_y)
      M3: Z exogenous -> X=f(Z,u_x); Y=f(X,u_y)
      M4: X exogenous -> Z=f(X,u_z); Y=f(X,u_y)
      M5: X exogenous -> Z=f(X,u_z); Y=f(X,Z,u_y)
      M6: X exogenous -> Y=f(X,u_y); Z=f(X,Y,u_z)
      M7: X exogenous -> Y=f(X,u_y); Z=f(Y,u_z)
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

    # Disturbances
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

    # Z for M1/M2/M3 (exogenous)
    if model_type in ['M1', 'M2', 'M3']:
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:
            z = z_base.astype(float)

    # X
    if model_type in ['M1', 'M3']:
        if case in ['a', 'c']:
            x = lam_zx * z + u_x
        else:
            if case == 'b':
                z_binary = (z > np.median(z)).astype(float)
            else:
                z_binary = z.copy()
            p_x = _logit_inv(lam_zx * (2 * z_binary - 1) - 0.5)
            x = np.random.binomial(1, p_x).astype(float)
    else:
        x = u_x.astype(float) if case in ['c', 'd'] else u_x

    # Z for M4/M5 (after X, before Y)
    if model_type in ['M4', 'M5']:
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:
            p_z = _logit_inv(lam_xz * (2 * x - 1) + 0.5 * z_base)
            z = np.random.binomial(1, p_z).astype(float)

    # Observed Y (for the centering factor)
    if model_type in ['M1', 'M2', 'M5']:
        y_eta = intercept + lam_xy * x + lam_zy * z + u_y
    else:
        y_eta = intercept + lam_xy * x + u_y
    centering = np.mean(y_eta)

    # Total-effect potential outcomes
    if model_type == 'M5':
        if case in ['a', 'c']:
            z1 = lam_xz * 1 + u_z
            z0 = lam_xz * 0 + u_z
        else:
            z1 = np.random.binomial(1, _logit_inv(lam_xz * (2 * 1 - 1) + 0.5 * z_base), n).astype(float)
            z0 = np.random.binomial(1, _logit_inv(lam_xz * (2 * 0 - 1) + 0.5 * z_base), n).astype(float)
        eta1 = intercept + lam_xy * 1 + lam_zy * z1 + u_y - centering
        eta0 = intercept + lam_xy * 0 + lam_zy * z0 + u_y - centering
    elif model_type in ['M1', 'M2']:
        eta1 = intercept + lam_xy * 1 + lam_zy * z + u_y - centering
        eta0 = intercept + lam_xy * 0 + lam_zy * z + u_y - centering
    else:
        eta1 = intercept + lam_xy * 1 + u_y - centering
        eta0 = intercept + lam_xy * 0 + u_y - centering
    return float(np.mean(_logit_inv(eta1) - _logit_inv(eta0)))


def get_true_direct_effect_case(model_type: str, case: str = 'a',
                                params: Optional[Dict] = None,
                                n_mc: int = 200000, seed: int = 42) -> float:
    """
    Compute the direct effect at the observed Z distribution.

    For M5 this is the target of the adjusted estimator (marginal standardization
    over the observed mediator distribution). For all other models Z is not on
    the X -> Y causal path (or Y is independent of Z), so direct == total.
    """
    if model_type != 'M5':
        return get_true_ace_case(model_type, case, params, n_mc, seed)

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
    lam_zy = model_params['lambda_zy']
    lam_xy = model_params['lambda_xy']
    lam_xz = model_params['lambda_xz']
    intercept = model_params['intercept']
    n = n_mc

    if case in ['a', 'b']:
        u_z = np.random.normal(0, 1, n)
        u_x = np.random.normal(0, 1, n)
        u_y = np.random.normal(0, 1, n)
        z_base = np.random.normal(0, 1, n)
    else:
        u_z = np.random.binomial(1, 0.5, n)
        u_x = np.random.binomial(1, 0.5, n)
        u_y = np.random.binomial(1, 0.5, n)
        z_base = np.random.binomial(1, 0.5, n)

    x = u_x.astype(float) if case in ['c', 'd'] else u_x
    if case in ['a', 'c']:
        z = lam_xz * x + u_z
        z = z - np.mean(z)
    else:
        p_z = _logit_inv(lam_xz * (2 * x - 1) + 0.5 * z_base)
        z = np.random.binomial(1, p_z).astype(float)

    y_eta = intercept + lam_xy * x + lam_zy * z + u_y
    centering = np.mean(y_eta)
    eta1 = intercept + lam_xy * 1 + lam_zy * z + u_y - centering
    eta0 = intercept + lam_xy * 0 + lam_zy * z + u_y - centering
    return float(np.mean(_logit_inv(eta1) - _logit_inv(eta0)))


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
