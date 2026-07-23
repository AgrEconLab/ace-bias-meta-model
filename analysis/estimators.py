"""
DREAM Pipeline — Estimator Suite for ACE (Average Causal Effect)

Implements:
  1. logit_ace_gcomp    — Logit + g-computation (marginal standardization)
  2. logit_ace_naive    — Logit + naive average marginal effect
  3. probit_ace_gcomp   — Probit + g-computation
  4. tmle_ace           — Targeted Maximum Likelihood Estimation
  5. ipw_ace            — Inverse Probability Weighting
  6. aipw_ace           — Doubly Robust AIPW

All estimators return dict: { 'ace': float, 'se': float, 'ci_lower': float, 'ci_upper': float }
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import CONFIG


# ─── Helper ───────────────────────────────────────────────────────────────────

def _logit(x: np.ndarray) -> np.ndarray:
    """Logit function (inverse of logistic)."""
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return np.log(x / (1 - x))


def _logistic(eta: np.ndarray) -> np.ndarray:
    """Logistic (inverse logit) function."""
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))


def _bootstrap_se(data: pd.DataFrame, estimator_fn, n_boot: int = 200,
                  alpha: float = 0.05) -> dict:
    """
    Bootstrap standard error for any estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset
    estimator_fn : callable
        Function that takes a DataFrame and returns ACE estimate
    n_boot : int
        Number of bootstrap replicates
    alpha : float
        Significance level for CI

    Returns
    -------
    dict with ace, se, ci_lower, ci_upper
    """
    n = len(data)
    ace_hat = estimator_fn(data)

    boot_estimates = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[idx]
        try:
            boot_estimates[b] = estimator_fn(boot_data)
        except Exception:
            boot_estimates[b] = np.nan

    boot_estimates = boot_estimates[~np.isnan(boot_estimates)]
    se = np.std(boot_estimates, ddof=1)
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    return {
        'ace': ace_hat,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 1: Logit + G-computation (Marginal Standardization)
# ═══════════════════════════════════════════════════════════════════════════════

def logit_ace_gcomp(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
                    z_name: str = 'Z', bootstrap: bool = True,
                    n_boot: int = 200) -> dict:
    """
    ACE via Logit + g-computation (marginal standardization).

    Procedure:
      1. Fit logit model: Y ~ X + Z
      2. For each observation, predict P(Y=1|X=1, Z=z_i) and P(Y=1|X=0, Z=z_i)
      3. ACE = average difference in predictions

    This is the theoretically correct implementation (causal-skillkit
    g-computation formula).
    """
    def _estimate(df):
        df = df.copy()

        # Fit logit model (with or without Z)
        if z_name:
            formula = f"{y_name} ~ {x_name} + {z_name}"
        else:
            formula = f"{y_name} ~ {x_name}"
        model = smf.logit(formula=formula, data=df).fit(disp=False, maxiter=100)

        # Predict under X=1 and X=0 (interventional)
        df_x1 = df.copy()
        df_x1[x_name] = 1.0
        p_y1 = model.predict(df_x1)

        df_x0 = df.copy()
        df_x0[x_name] = 0.0
        p_y0 = model.predict(df_x0)

        ace = float(np.mean(p_y1 - p_y0))
        return ace

    if bootstrap:
        return _bootstrap_se(data, _estimate, n_boot=n_boot)
    else:
        ace = _estimate(data)
        return {'ace': ace, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 2: Logit + Naive Average Marginal Effect
# ═══════════════════════════════════════════════════════════════════════════════

def logit_ace_naive(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
                    z_name: str = 'Z', bootstrap: bool = True,
                    n_boot: int = 200) -> dict:
    """
    Naive ACE: coefficient-based interpretation (common but incorrect).

    Uses the logit marginal effect at means approach:
    ACE ≈ β_X * P(Y=1|X=1) * (1 - P(Y=1|X=1))

    NOTE: This is included for COMPARISON only. It is NOT the recommended
    approach for non-linear models.
    """
    def _estimate(df):
        df = df.copy()

        # Fit logit (with or without Z)
        if z_name:
            formula = f"{y_name} ~ {x_name} + {z_name}"
        else:
            formula = f"{y_name} ~ {x_name}"
        model = smf.logit(formula=formula, data=df).fit(disp=False, maxiter=100)

        # Average marginal effect of X
        beta_x = model.params[x_name]
        p_pred = model.predict(df)
        me_at_each = beta_x * p_pred * (1 - p_pred)
        ace = float(np.mean(me_at_each))
        return ace

    if bootstrap:
        return _bootstrap_se(data, _estimate, n_boot=n_boot)
    else:
        ace = _estimate(data)
        return {'ace': ace, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 3: Probit + G-computation
# ═══════════════════════════════════════════════════════════════════════════════

def probit_ace_gcomp(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
                     z_name: str = 'Z', bootstrap: bool = True,
                     n_boot: int = 200) -> dict:
    """
    ACE via Probit + g-computation.

    Same as logit_ace_gcomp but with probit link.
    Used to test sensitivity of findings to link function choice.
    """
    def _estimate(df):
        df = df.copy()
        if z_name:
            formula = f"{y_name} ~ {x_name} + {z_name}"
        else:
            formula = f"{y_name} ~ {x_name}"
        model = smf.probit(formula=formula, data=df).fit(disp=False, maxiter=100)

        df_x1 = df.copy()
        df_x1[x_name] = 1.0
        p_y1 = model.predict(df_x1)

        df_x0 = df.copy()
        df_x0[x_name] = 0.0
        p_y0 = model.predict(df_x0)

        ace = float(np.mean(p_y1 - p_y0))
        return ace

    if bootstrap:
        return _bootstrap_se(data, _estimate, n_boot=n_boot)
    else:
        ace = _estimate(data)
        return {'ace': ace, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 4: TMLE (Targeted Maximum Likelihood Estimation)
# ═══════════════════════════════════════════════════════════════════════════════

def tmle_ace(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
             z_name: str = 'Z', bootstrap: bool = False,
             q_formula: str = None) -> dict:
    """
    TMLE for ACE with binary outcome and binary/continuous treatment.

    Standard TMLE-ATE algorithm:
      1. Q(0): Initial outcome regression E[Y|A,W] via logit (with X*Z interaction)
      2. g(W): Propensity score model P(A|W) via logit
      3. H(A,W): Clever covariate = I(A=1)/g(W) - I(A=0)/(1-g(W))
      4. Fluctuation: logit(Y) = offset(logit(Q0)) + epsilon * H
      5. TMLE prediction under A=1 and A=0
      6. ACE = mean(Q1_1 - Q1_0)

    Parameters
    ----------
    q_formula : str, optional
        Custom formula for the initial Q model. If None, uses 'Y ~ X * Z'.
        Use this to test TMLE under model misspecification (e.g., 'Y ~ X'
        to omit Z, or 'Y ~ X + Z' to omit interaction).

    Reference: van der Laan & Rose (2011, 2018)
    """
    df = data.copy()
    n = len(df)

    # ── Step 1: Initial outcome regression Q(0) ──
    if q_formula is None:
        formula_q = f"{y_name} ~ {x_name} * {z_name}"
    else:
        formula_q = q_formula
    try:
        q_model = smf.logit(formula=formula_q, data=df).fit(disp=False, maxiter=100)
    except Exception:
        # Fall back to main effects
        q_model = smf.logit(formula=f"{y_name} ~ {x_name} + {z_name}",
                            data=df).fit(disp=False, maxiter=100)

    q1_pred = q_model.predict(df.assign(**{x_name: 1.0}))
    q0_pred = q_model.predict(df.assign(**{x_name: 0.0}))

    # Clamp to avoid log(0)
    q1_pred = np.clip(q1_pred, 1e-8, 1 - 1e-8)
    q0_pred = np.clip(q0_pred, 1e-8, 1 - 1e-8)

    # ── Step 2: Propensity score g(W) ──
    formula_g = f"{x_name} ~ {z_name}"
    try:
        if df[x_name].nunique() == 2:
            g_model = smf.logit(formula=formula_g, data=df).fit(disp=False, maxiter=100)
            g_pred = g_model.predict(df)
        else:
            g_model = sm.OLS.from_formula(formula_g, data=df).fit()
            g_pred = g_model.predict(df)
            g_pred = np.clip(g_pred, 0.01, 0.99)
    except Exception:
        # Fall back to constant propensity
        g_pred = np.ones(n) * df[x_name].mean()
        g_pred = np.clip(g_pred, 0.01, 0.99)

    a = df[x_name].values.astype(float)
    y = df[y_name].values.astype(float)

    # ── Step 3: Clever covariate H(A,W) ──
    h = a / g_pred - (1 - a) / (1 - g_pred)

    # ── Step 4: Fluctuation step ──
    # Predicted values under observed treatment
    q_obs_pred = q1_pred * a + q0_pred * (1 - a)
    q_obs_logit = _logit(np.clip(q_obs_pred, 1e-8, 1 - 1e-8))

    # Fluctuation regression: logit(Y) ~ offset(logit(Q0)) + epsilon * H (NO intercept)
    # Standard TMLE-ATE: only H as regressor, intercept is absorbed by offset
    try:
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Binomial

        fluct_model = GLM(
            y, h,
            offset=q_obs_logit,
            family=Binomial()
        ).fit(disp=False, maxiter=100)

        epsilon = float(fluct_model.params[0])
    except Exception:
        epsilon = 0.0

    # ── Step 5: Updated predictions ──
    q1_logit = _logit(np.clip(q1_pred, 1e-8, 1 - 1e-8))
    q0_logit = _logit(np.clip(q0_pred, 1e-8, 1 - 1e-8))

    q1_star = _logistic(q1_logit + epsilon / np.clip(g_pred, 0.01, 0.99))
    q0_star = _logistic(q0_logit - epsilon / np.clip(1 - g_pred, 0.01, 0.99))

    ace = float(np.mean(q1_star - q0_star))

    # ── Variance estimation ──
    # IC-based variance
    ic = (a / g_pred * (y - q1_star) - (1 - a) / (1 - g_pred) * (y - q0_star)
          + (q1_star - q0_star) - ace)
    se = float(np.std(ic, ddof=1) / np.sqrt(n))
    ci_lower = ace - 1.96 * se
    ci_upper = ace + 1.96 * se

    return {
        'ace': ace,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'epsilon': epsilon,  # Diagnostic: should be near 0 if Q0 is well-specified
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 5: IPW (Inverse Probability Weighting)
# ═══════════════════════════════════════════════════════════════════════════════

def ipw_ace(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
            z_name: str = 'Z', bootstrap: bool = True, n_boot: int = 200) -> dict:
    """
    ACE via Inverse Probability Weighting.

    Procedure:
      1. Estimate propensity score P(A=1|W) via logit
      2. Weight outcomes by inverse propensity
      3. ACE = E[Y*A/g(W)] - E[Y*(1-A)/(1-g(W))]

    NOTE: Only valid for binary treatment (X). Returns NaN for continuous X.
    """
    def _estimate(df):
        df = df.copy()
        n = len(df)

        # IPW requires binary treatment
        if df[x_name].nunique() > 2:
            return np.nan

        a = df[x_name].values.astype(float)
        y = df[y_name].values.astype(float)

        # Propensity score
        formula_g = f"{x_name} ~ {z_name}"
        if df[x_name].nunique() == 2:
            g_model = smf.logit(formula=formula_g, data=df).fit(disp=False, maxiter=100)
            g_pred = g_model.predict(df)
        else:
            g_model = sm.OLS.from_formula(formula_g, data=df).fit()
            g_pred = np.clip(g_model.predict(df), 0.01, 0.99)

        g_pred = np.clip(g_pred, 0.01, 0.99)

        # IPW estimator
        w1 = a / g_pred
        w0 = (1 - a) / (1 - g_pred)

        ace = float(np.mean(w1 * y) / np.mean(w1) - np.mean(w0 * y) / np.mean(w0))
        return ace

    if bootstrap:
        return _bootstrap_se(data, _estimate, n_boot=n_boot)
    else:
        ace = _estimate(data)
        return {'ace': ace, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 6: AIPW (Doubly Robust Augmented IPW)
# ═══════════════════════════════════════════════════════════════════════════════

def aipw_ace(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
             z_name: str = 'Z', bootstrap: bool = True, n_boot: int = 200) -> dict:
    """
    ACE via Doubly Robust AIPW.

    Combines outcome regression + propensity score.
    Consistent if EITHER the outcome model OR the propensity model is correct.

    ACE = 1/n Σ [Â(1,W_i) - Â(0,W_i)
          + A_i/ĝ(W_i) (Y_i - Â(1,W_i))
          - (1-A_i)/(1-ĝ(W_i)) (Y_i - Â(0,W_i))]

    NOTE: Only valid for binary treatment (X). Returns NaN for continuous X.
    """
    def _estimate(df):
        df = df.copy()
        n = len(df)

        # AIPW requires binary treatment
        if df[x_name].nunique() > 2:
            return np.nan

        a = df[x_name].values.astype(float)
        y = df[y_name].values.astype(float)

        # Outcome model Q
        try:
            q_formula = f"{y_name} ~ {x_name} * {z_name}"
            q_model = smf.logit(formula=q_formula, data=df).fit(disp=False, maxiter=100)
        except Exception:
            q_formula = f"{y_name} ~ {x_name} + {z_name}"
            q_model = smf.logit(formula=q_formula, data=df).fit(disp=False, maxiter=100)

        q1 = q_model.predict(df.assign(**{x_name: 1.0}))
        q0 = q_model.predict(df.assign(**{x_name: 0.0}))

        # Propensity score
        g_formula = f"{x_name} ~ {z_name}"
        if df[x_name].nunique() == 2:
            g_model = smf.logit(formula=g_formula, data=df).fit(disp=False, maxiter=100)
            g_pred = g_model.predict(df)
        else:
            g_model = sm.OLS.from_formula(g_formula, data=df).fit()
            g_pred = np.clip(g_model.predict(df), 0.01, 0.99)

        g_pred = np.clip(g_pred, 0.01, 0.99)

        # AIPW
        aipw_1 = q1 + a / g_pred * (y - q1)
        aipw_0 = q0 + (1 - a) / (1 - g_pred) * (y - q0)

        ace = float(np.mean(aipw_1 - aipw_0))
        return ace

    if bootstrap:
        return _bootstrap_se(data, _estimate, n_boot=n_boot)
    else:
        ace = _estimate(data)
        return {'ace': ace, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator 7: LPM (Linear Probability Model) via OLS + G-computation
# ═══════════════════════════════════════════════════════════════════════════════

def lpm_ace_ols(data: pd.DataFrame, y_name: str = 'Y', x_name: str = 'X',
                z_name: str = 'Z', bootstrap: bool = True,
                n_boot: int = 200) -> dict:
    """
    ACE via Linear Probability Model (OLS) + g-computation.

    Fits OLS(Y ~ X + Z) with HC3 robust standard errors, then computes ACE
    via g-computation: ACE = 1/n Σ [E[Y|X=1, Z_i] - E[Y|X=0, Z_i]].

    The LPM coefficient on X directly estimates the ACE on probability scale,
    and g-computation yields the same result. This approach is consistent
    with the logit_gcomp interface for parallel comparison.

    LPM avoids non-collapsibility (unlike logit OR) but is technically
    misspecified for binary outcomes (predictions may fall outside [0,1]).
    """
    def _estimate(df):
        df = df.copy()
        if z_name:
            formula = f"{y_name} ~ {x_name} + {z_name}"
        else:
            formula = f"{y_name} ~ {x_name}"
        model = sm.OLS.from_formula(formula, data=df).fit(cov_type='HC3')

        # G-computation: predict under X=1 and X=0
        df_x1 = df.copy()
        df_x1[x_name] = 1.0
        p_y1 = model.predict(df_x1)

        df_x0 = df.copy()
        df_x0[x_name] = 0.0
        p_y0 = model.predict(df_x0)

        ace = float(np.mean(p_y1 - p_y0))
        return ace

    if bootstrap:
        return _bootstrap_se(data, _estimate, n_boot=n_boot)
    else:
        ace = _estimate(data)
        # HC3 SE from the model for diagnostic purposes
        df = data.copy()
        if z_name:
            formula = f"{y_name} ~ {x_name} + {z_name}"
        else:
            formula = f"{y_name} ~ {x_name}"
        model = sm.OLS.from_formula(formula, data=df).fit(cov_type='HC3')
        return {
            'ace': ace,
            'se': float(model.bse[x_name]),
            'ci_lower': ace - 1.96 * float(model.bse[x_name]),
            'ci_upper': ace + 1.96 * float(model.bse[x_name]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator Registry
# ═══════════════════════════════════════════════════════════════════════════════

ESTIMATOR_REGISTRY = {
    'logit_ace_gcomp': logit_ace_gcomp,
    'logit_ace_naive': logit_ace_naive,
    'probit_ace_gcomp': probit_ace_gcomp,
    'tmle_ace': tmle_ace,
    'ipw_ace': ipw_ace,
    'aipw_ace': aipw_ace,
    'lpm_ace_ols': lpm_ace_ols,
}


def estimate_all(data: pd.DataFrame, estimators: list = None,
                 y_name: str = 'Y', x_name: str = 'X', z_name: str = 'Z',
                 bootstrap: bool = True, n_boot: int = 200,
                 tmle_q_formula: str = None) -> dict:
    """
    Run all specified estimators on a dataset.

    Parameters
    ----------
    data : pd.DataFrame
    estimators : list of str, optional
        Defaults to CONFIG.estimators
    y_name, x_name, z_name : str
        Variable names
    bootstrap : bool
        Whether to bootstrap SE (not used for TMLE which has IC-based SE)
    n_boot : int
        Number of bootstrap replicates

    Returns
    -------
    dict mapping estimator_name -> result dict
    """
    if estimators is None:
        estimators = CONFIG.estimators

    results = {}
    for est_name in estimators:
        if est_name not in ESTIMATOR_REGISTRY:
            continue

        try:
            est_fn = ESTIMATOR_REGISTRY[est_name]
            # TMLE doesn't need bootstrap (has IC-based SE)
            if est_name == 'tmle_ace':
                result = est_fn(data, y_name=y_name, x_name=x_name, z_name=z_name,
                               bootstrap=False, q_formula=tmle_q_formula)
            else:
                result = est_fn(data, y_name=y_name, x_name=x_name, z_name=z_name,
                               bootstrap=bootstrap, n_boot=n_boot)
            results[est_name] = result
        except Exception as e:
            results[est_name] = {
                'ace': np.nan, 'se': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'error': str(e)
            }

    return results
