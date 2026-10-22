"""
Multi-covariate extension for the meta-model paper.

Tests whether an RF trained on single-covariate DAGs (M1/M5) generalizes to
multi-covariate settings with 2 covariates per DAG.

DAGs:
  M8:  Confounder Z1 + Irrelevant Z2  (adjusting both = correct)
  M9:  Confounder Z1 + Collider Z2    (adjusting both = collider bias)
  M10: Two Confounders Z1, Z2         (adjusting both = correct)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import time, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

SEED = 20240625

# ── Helpers ──────────────────────────────────────────────────────────────────

def _logit_inv(x):
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

TREE_FEATURES = ['lpm_x_coef', 'n']

Z_NAMES = ['Z1', 'Z2']
Z_FORMULA = '+'.join(Z_NAMES)       # "Z1+Z2"


# ═══════════════════════════════════════════════════════════════════════════
# PART A: Multi-Covariate DGPs
# ═══════════════════════════════════════════════════════════════════════════

def _z_exogenous(n, case, rs):
    """Generate exogenous Z (follows M1's Z generation)."""
    if case in ['a', 'c']:
        z_base = rs.normal(0, 1, n)
        z = 0.5 * z_base + 0.5 * rs.normal(0, 1, n)
    else:
        z_base = rs.binomial(1, 0.5, n)
        z = z_base.astype(float)
    return z


def _generate_x_from_z(z, lam_zx, case, n, rs):
    """Generate X from Z (follows M1's X generation). Cases a/d only."""
    if case == 'a':
        return lam_zx * z + rs.normal(0, 1, n)
    else:  # 'd' — Bernoulli X from logit model
        p_x = _logit_inv(lam_zx * (2 * z - 1) - 0.5)
        return rs.binomial(1, p_x).astype(float)


def generate_m8_data(n, case, params, rs):
    """M8: Confounder Z1 + Irrelevant Z2. Adjust both = correct."""
    lam_zx = params.get('lambda_zx', 0.3)
    lam_zy = params.get('lambda_zy', 0.3)
    lam_xy = params.get('lambda_xy', 0.1)
    intercept = params.get('intercept', 0.0)

    u_y = rs.normal(0, 1, n) if case == 'a' else rs.binomial(1, 0.5, n)

    z1 = _z_exogenous(n, case, rs)
    z2 = _z_exogenous(n, case, rs)  # independent, irrelevant
    x = _generate_x_from_z(z1, lam_zx, case, n, rs)

    y_eta = intercept + lam_xy * x + lam_zy * z1 + u_y  # Z2 absent
    y_eta = y_eta - np.mean(y_eta)
    y = rs.binomial(1, _logit_inv(y_eta)).astype(float)

    return pd.DataFrame({'Y': y, 'X': x, 'Z1': z1, 'Z2': z2})


def generate_m9_data(n, case, params, rs):
    """M9: Confounder Z1 + Collider Z2. Adjust both = opens collider."""
    lam_zx = params.get('lambda_zx', 0.3)
    lam_zy = params.get('lambda_zy', 0.3)
    lam_xy = params.get('lambda_xy', 0.1)
    lam_xz = params.get('lambda_xz', 0.3)
    lam_yz = params.get('lambda_yz', 0.3)
    intercept = params.get('intercept', 0.0)

    u_y = rs.normal(0, 1, n) if case == 'a' else rs.binomial(1, 0.5, n)
    u_z2 = rs.normal(0, 1, n) if case == 'a' else rs.binomial(1, 0.5, n)

    z1 = _z_exogenous(n, case, rs)                     # confounder
    x = _generate_x_from_z(z1, lam_zx, case, n, rs)       # X depends on Z1

    y_eta = intercept + lam_xy * x + lam_zy * z1 + u_y # Y depends on X + Z1
    y_eta = y_eta - np.mean(y_eta)
    y = rs.binomial(1, _logit_inv(y_eta)).astype(float)

    # Z2 is a collider: generated after X and Y
    if case == 'a':
        z2 = 0.5 * (lam_xz * x + lam_yz * y) + 0.5 * u_z2
    else:
        p_z2 = _logit_inv(lam_xz * (2 * x - 1) + lam_yz * (2 * y - 1) + 0.5 * u_z2)
        z2 = rs.binomial(1, p_z2).astype(float)

    return pd.DataFrame({'Y': y, 'X': x, 'Z1': z1, 'Z2': z2})


def generate_m10_data(n, case, params, rs):
    """M10: Two Confounders Z1, Z2. Adjust both = correct."""
    lam_zx1 = params.get('lambda_zx', 0.3)
    lam_zx2 = params.get('lambda_zx2', params.get('lambda_zx', 0.3))
    lam_zy1 = params.get('lambda_zy', 0.3)
    lam_zy2 = params.get('lambda_zy2', params.get('lambda_zy', 0.3))
    lam_xy = params.get('lambda_xy', 0.1)
    intercept = params.get('intercept', 0.0)

    u_y = rs.normal(0, 1, n) if case == 'a' else rs.binomial(1, 0.5, n)

    z1 = _z_exogenous(n, case, rs)
    z2 = _z_exogenous(n, case, rs)

    # X depends on both confounders
    if case == 'a':
        x = lam_zx1 * z1 + lam_zx2 * z2 + rs.normal(0, 1, n)
    else:
        p_x = _logit_inv(lam_zx1 * (2 * z1 - 1) + lam_zx2 * (2 * z2 - 1) - 0.5)
        x = rs.binomial(1, p_x).astype(float)

    # Y depends on X + both confounders
    y_eta = intercept + lam_xy * x + lam_zy1 * z1 + lam_zy2 * z2 + u_y
    y_eta = y_eta - np.mean(y_eta)
    y = rs.binomial(1, _logit_inv(y_eta)).astype(float)

    return pd.DataFrame({'Y': y, 'X': x, 'Z1': z1, 'Z2': z2})


# Registry
MULTICOV_DGPS = {
    'M8': generate_m8_data,
    'M9': generate_m9_data,
    'M10': generate_m10_data,
}


# ═══════════════════════════════════════════════════════════════════════════
# PART B: True ACE (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════

def true_ace_m8(params, case='a', n_mc=200000, seed=42):
    """True ACE for M8. Y depends on X and Z1 (not Z2).

    The centering constant is case-specific: case a draws continuous normal
    variables, case d draws binary Bernoulli variables. Reusing the case-a
    truth for case d shifts the estimand.
    """
    rs = np.random.RandomState(seed)
    lam_zx = params.get('lambda_zx', 0.3)
    lam_zy = params.get('lambda_zy', 0.3)
    lam_xy = params.get('lambda_xy', 0.1)
    intercept = params.get('intercept', 0.0)

    if case == 'a':
        u_y = rs.normal(0, 1, n_mc)
        z1 = _z_exogenous(n_mc, 'a', rs)
        z2 = _z_exogenous(n_mc, 'a', rs)  # not used in Y
        x = lam_zx * z1 + rs.normal(0, 1, n_mc)
    elif case == 'd':
        u_y = rs.binomial(1, 0.5, n_mc)
        z1 = rs.binomial(1, 0.5, n_mc).astype(float)
        z2 = rs.binomial(1, 0.5, n_mc).astype(float)  # not used in Y
        p_x = _logit_inv(lam_zx * (2 * z1 - 1) - 0.5)
        x = rs.binomial(1, p_x, n_mc).astype(float)
    else:
        raise ValueError(f"unsupported case: {case}")

    y_eta = intercept + lam_xy * x + lam_zy * z1 + u_y
    centering = np.mean(y_eta)

    eta1 = intercept + lam_xy * 1 + lam_zy * z1 + u_y - centering
    eta0 = intercept + lam_xy * 0 + lam_zy * z1 + u_y - centering
    ace = float(np.mean(_logit_inv(eta1) - _logit_inv(eta0)))
    return ace


def true_ace_m9(params, case='a', n_mc=200000, seed=42):
    """True ACE for M9. Same Y equation as M8 (collider Z2 doesn't affect Y)."""
    return true_ace_m8(params, case=case, n_mc=n_mc, seed=seed)


def true_ace_m10(params, case='a', n_mc=200000, seed=42):
    """True ACE for M10. Y depends on X, Z1, Z2 — both confounders."""
    rs = np.random.RandomState(seed)
    lam_zx1 = params.get('lambda_zx', 0.3)
    lam_zx2 = params.get('lambda_zx2', params.get('lambda_zx', 0.3))
    lam_zy1 = params.get('lambda_zy', 0.3)
    lam_zy2 = params.get('lambda_zy2', params.get('lambda_zy', 0.3))
    lam_xy = params.get('lambda_xy', 0.1)
    intercept = params.get('intercept', 0.0)

    if case == 'a':
        u_y = rs.normal(0, 1, n_mc)
        z1 = _z_exogenous(n_mc, 'a', rs)
        z2 = _z_exogenous(n_mc, 'a', rs)
        x = lam_zx1 * z1 + lam_zx2 * z2 + rs.normal(0, 1, n_mc)
    elif case == 'd':
        u_y = rs.binomial(1, 0.5, n_mc)
        z1 = rs.binomial(1, 0.5, n_mc).astype(float)
        z2 = rs.binomial(1, 0.5, n_mc).astype(float)
        p_x = _logit_inv(lam_zx1 * (2 * z1 - 1)
                         + lam_zx2 * (2 * z2 - 1) - 0.5)
        x = rs.binomial(1, p_x, n_mc).astype(float)
    else:
        raise ValueError(f"unsupported case: {case}")

    y_eta = intercept + lam_xy * x + lam_zy1 * z1 + lam_zy2 * z2 + u_y
    centering = np.mean(y_eta)

    eta1 = intercept + lam_xy * 1 + lam_zy1 * z1 + lam_zy2 * z2 + u_y - centering
    eta0 = intercept + lam_xy * 0 + lam_zy1 * z1 + lam_zy2 * z2 + u_y - centering
    ace = float(np.mean(_logit_inv(eta1) - _logit_inv(eta0)))
    return ace


TRUE_ACE_FUNCS = {
    'M8': true_ace_m8,
    'M9': true_ace_m9,
    'M10': true_ace_m10,
}


# ═══════════════════════════════════════════════════════════════════════════
# PART C: Meta-Feature Computation (multi-covariate adaptation)
# ═══════════════════════════════════════════════════════════════════════════

def compute_meta_features(data, z_names, logit_ace_val, lpm_ace_val, unadj_ace_val):
    """Compute 16 meta-features from multi-covariate data.

    For features involving Z coefficients, we take max |coef| across all Z's.
    """
    n = len(data)
    y_mean = float(data['Y'].mean())
    x_var = float(data['X'].var(ddof=1))

    z_formula = '+'.join(z_names)

    # Logit model: Y ~ X + Z1 + Z2
    logit_model = sm.Logit.from_formula(
        f"Y ~ X + {z_formula}", data=data).fit(disp=False, maxiter=100)

    # z_coef_mag = max |coef| across all Z's
    z_coefs = [abs(logit_model.params.get(z, 0)) for z in z_names]
    z_coef_mag = float(max(z_coefs))

    # Predicted probabilities
    pred = logit_model.predict(data)
    y_pred_min = float(pred.min())
    y_pred_max = float(pred.max())

    # OLS X ~ Z1 + Z2 for Z→X association
    ols_zx = sm.OLS.from_formula(f"X ~ {z_formula}", data=data).fit()
    z_x_coefs = [abs(ols_zx.params.get(z, 0)) for z in z_names]
    z_x_coef_mag = float(max(z_x_coefs))
    z_x_r2 = float(ols_zx.rsquared)

    # LPM: Y ~ X + Z1 + Z2 with HC3
    lpm_model = sm.OLS.from_formula(
        f"Y ~ X + {z_formula}", data=data).fit(cov_type='HC3')
    lpm_x_coef = float(lpm_model.params.get('X', 0))

    # OLS Y ~ X + Z1 + Z2 for R2
    ols_y = sm.OLS.from_formula(
        f"Y ~ X + {z_formula}", data=data).fit()
    y_model_r2 = float(ols_y.rsquared)

    return {
        'n': n,
        'y_mean': y_mean,
        'x_var': x_var,
        'logit_ace': logit_ace_val,
        'lpm_ace': lpm_ace_val,
        'logit_lpm_diff': abs(logit_ace_val - lpm_ace_val),
        'unadj_ace': unadj_ace_val,
        'adj_unadj_diff': abs(logit_ace_val - unadj_ace_val),
        'z_coef_mag': z_coef_mag,
        'z_x_coef_mag': z_x_coef_mag,
        'z_x_r2': z_x_r2,
        'y_pred_min': y_pred_min,
        'y_pred_max': y_pred_max,
        'y_pred_range': y_pred_max - y_pred_min,
        'lpm_x_coef': lpm_x_coef,
        'y_model_r2': y_model_r2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART D: Parameter Grid
# ═══════════════════════════════════════════════════════════════════════════

LAMBDA_XY = [0.05, 0.10, 0.30]
SAMPLE_SIZES = [250, 1000, 5000]
CASES = ['a', 'd']  # match existing meta_generate extremes
N_ITER = 20

MULTICOV_PARAM_GRID = {
    'M8': {
        'vary': ['lambda_zx', 'lambda_zy'],
        'values': [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5)],
    },
    'M9': {
        'vary': ['lambda_xz', 'lambda_yz'],
        'values': [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5)],
    },
    'M10': {
        'vary': ['lambda_zx', 'lambda_zy'],
        'values': [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5)],
        'extra': {'lambda_zx2': None, 'lambda_zy2': None},  # same values for both confounders
    },
}


def generate_configs():
    """Yield (model, case, n, params) for all grid cells."""
    configs = []
    for model, spec in MULTICOV_PARAM_GRID.items():
        for case in CASES:
            for lam_xy in LAMBDA_XY:
                for n in SAMPLE_SIZES:
                    for vals in spec['values']:
                        params = {'lambda_xy': lam_xy}
                        for i, varname in enumerate(spec['vary']):
                            params[varname] = vals[i]
                        # For M10, set both confounders to same value
                        if model == 'M10':
                            params['lambda_zx2'] = params['lambda_zx']
                            params['lambda_zy2'] = params['lambda_zy']
                        configs.append((model, case, n, params))
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# PART E: Simulation Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(configs):
    """Run all multi-covariate simulations and return results DataFrame."""
    records = []
    n_config = len(configs)
    total_sims = n_config * N_ITER
    sim_start = time.time()

    print(f"\nGenerating {n_config} configs x {N_ITER} iters = {total_sims} simulations")
    print(f"{'─' * 60}")

    # Precompute true ACE for each unique param set
    true_ace_cache = {}
    for model, case, n, params in configs:
        key = (model, case, tuple(sorted(params.items())))
        if key not in true_ace_cache:
            t0 = time.time()
            ace = TRUE_ACE_FUNCS[model](params, case=case)
            true_ace_cache[key] = ace
    print(f"  Precomputed {len(true_ace_cache)} true ACE values "
          f"({time.time() - sim_start:.0f}s)")

    for cfg_idx, (model, case, n, params) in enumerate(configs):
        if (cfg_idx + 1) % 10 == 0:
            elapsed = time.time() - sim_start
            rate = (cfg_idx + 1) * N_ITER / elapsed if elapsed > 0 else 0
            pct = (cfg_idx + 1) / n_config * 100
            print(f"  [{pct:.0f}%] config {cfg_idx+1}/{n_config} "
                  f"({rate:.0f} sim/s, {elapsed:.0f}s)")

        key = (model, case, tuple(sorted(params.items())))
        true_ace = true_ace_cache[key]
        gen_func = MULTICOV_DGPS[model]

        for it in range(N_ITER):
            seed = SEED + cfg_idx + it * 10000
            rs = np.random.RandomState(seed)
            data = gen_func(n, case, params, rs)

            # ACE estimates
            logit_model = sm.Logit.from_formula(
                f"Y ~ X + {Z_FORMULA}", data=data).fit(disp=False, maxiter=100)
            df_x1, df_x0 = data.copy(), data.copy()
            df_x1['X'] = 1.0
            df_x0['X'] = 0.0
            logit_ace = float(np.mean(logit_model.predict(df_x1) -
                                       logit_model.predict(df_x0)))

            lpm_model = sm.OLS.from_formula(
                f"Y ~ X + {Z_FORMULA}", data=data).fit(cov_type='HC3')
            df_x1['X'] = 1.0
            df_x0['X'] = 0.0
            # LPM g-computation using OLS predictions
            lpm_pred1 = lpm_model.predict(df_x1)
            lpm_pred0 = lpm_model.predict(df_x0)
            lpm_ace = float(np.mean(lpm_pred1 - lpm_pred0))

            unadj_model = sm.Logit.from_formula(
                "Y ~ X", data=data).fit(disp=False, maxiter=100)
            df_x1['X'] = 1.0
            df_x0['X'] = 0.0
            unadj_ace = float(np.mean(unadj_model.predict(df_x1) -
                                       unadj_model.predict(df_x0)))

            # Meta-features
            mf = compute_meta_features(data, Z_NAMES,
                                       logit_ace, lpm_ace, unadj_ace)

            # Bias
            logit_bias_pct = (logit_ace - true_ace) / abs(true_ace) * 100

            record = {**mf,
                      'true_ace': true_ace,
                      'abs_bias_pct': abs(logit_bias_pct),
                      'high_bias': int(abs(logit_bias_pct) > 5),
                      'model_type': model,
                      'case_type': case,
                      'lambda_xy': params.get('lambda_xy'),
                      }
            # Add varying params
            for k, v in params.items():
                if k != 'lambda_xy':
                    record[f'param_{k}'] = v
            records.append(record)

    elapsed = time.time() - sim_start
    print(f"\n  Done: {len(records)} records in {elapsed:.0f}s "
          f"({len(records)/elapsed:.0f} sim/s)")
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# PART F: Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def prepare(X_df, y):
    """Clean features."""
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def evaluate_rf_on_multicov(rf, multicov_df, label):
    """Apply a trained RF to multi-covariate data and report metrics."""
    X_mc, y_mc = prepare(multicov_df[FEATURES],
                          multicov_df['abs_bias_pct'].values)
    y_pred = rf.predict(X_mc)

    r2 = r2_score(y_mc, y_pred)
    mae = mean_absolute_error(y_mc, y_pred)

    # Per-DAG
    per_dag = {}
    multicov_df = multicov_df.copy()
    multicov_df['pred'] = y_pred
    for model in sorted(multicov_df['model_type'].unique()):
        sub = multicov_df[multicov_df['model_type'] == model]
        r2_m = r2_score(sub['abs_bias_pct'], sub['pred'])
        mae_m = mean_absolute_error(sub['abs_bias_pct'], sub['pred'])
        per_dag[model] = {'r2': r2_m, 'mae': mae_m, 'n': len(sub)}

    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(f"  R2 = {r2:.4f} | MAE = {mae:.2f}pp")
    for m in sorted(per_dag.keys()):
        print(f"  {m}: R2={per_dag[m]['r2']:.4f}, MAE={per_dag[m]['mae']:.2f}pp "
              f"(n={per_dag[m]['n']})")

    return {'r2': r2, 'mae': mae, 'per_dag': per_dag}


def train_rf_single(X_train, y_train):
    """Train RF on single-covariate training data. Same params as meta_train."""
    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    return rf


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("MULTI-COVARIATE EXTENSION")
    print("=" * 60)

    # ── 1. Load single-covariate training data ──
    base = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base, '../results/raw/meta_training_data.parquet')
    df_single = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df_single)} single-covariate rows")

    # ── 2. Train RF on single-covariate M1/M5 ──
    train_df = df_single[df_single['model_type'].isin(['M1', 'M5'])].copy()
    test_df_single = df_single[df_single['model_type'].isin(['M6', 'M7'])].copy()
    print(f"Train single: {len(train_df)} rows (M1/M5)")
    print(f"Test single (held-out): {len(test_df_single)} rows (M6/M7)")

    X_train, y_train = prepare(train_df[FEATURES], train_df['abs_bias_pct'].values)
    X_test_s, y_test_s = prepare(test_df_single[FEATURES],
                                  test_df_single['abs_bias_pct'].values)

    rf = train_rf_single(X_train, y_train)
    r2_single = r2_score(y_test_s, rf.predict(X_test_s))
    mae_single = mean_absolute_error(y_test_s, rf.predict(X_test_s))
    print(f"  RF R2 (single held-out M6/M7): {r2_single:.4f}")
    print(f"  RF MAE (single held-out M6/M7): {mae_single:.2f}pp")

    # ── 3. Generate multi-covariate data ──
    configs = generate_configs()
    print(f"\nMulti-covariate grid: {len(configs)} configs")
    multicov_df = run_simulation(configs)

    # Summary stats
    print(f"\nMulti-covariate data summary:")
    print(f"  Rows: {len(multicov_df)}")
    print(f"  High-bias rate: {multicov_df['high_bias'].mean()*100:.1f}%")
    print(f"  Mean |bias|: {multicov_df['abs_bias_pct'].mean():.1f}%")
    for model in ['M8', 'M9', 'M10']:
        sub = multicov_df[multicov_df['model_type'] == model]
        print(f"  {model}: n={len(sub)}, high-bias={sub['high_bias'].mean()*100:.0f}%, "
              f"mean |bias|={sub['abs_bias_pct'].mean():.1f}%")

    # ── 4. Apply single-covariate RF to multi-covariate data ──
    print(f"\n{'=' * 60}")
    print("GENERALIZATION: Single RF → Multi-covariate")
    print(f"{'=' * 60}")

    res_gen = evaluate_rf_on_multicov(rf, multicov_df,
                                       "Single RF → Multi-covariate (all)")

    # Per-DAG breakdown
    for model in ['M8', 'M9', 'M10']:
        sub = multicov_df[multicov_df['model_type'] == model]
        evaluate_rf_on_multicov(rf, sub, f"Single RF → {model}")

    # ── 5. Re-train on combined data ──
    print(f"\n{'=' * 60}")
    print("COMBINED: Train on Single + Multi, Test on Held-out Multi")
    print(f"{'=' * 60}")

    # Combine single M1/M5 + multi data for training
    multicov_features = multicov_df[FEATURES].copy()
    multicov_target = multicov_df['abs_bias_pct'].values

    X_combined = pd.concat([X_train, multicov_features], ignore_index=True)
    y_combined = np.concatenate([y_train, multicov_target])

    rf_combined = train_rf_single(X_combined, y_combined)
    r2_comb_single = r2_score(y_test_s, rf_combined.predict(X_test_s))
    print(f"  RF R2 on single held-out (M6/M7): {r2_comb_single:.4f}")

    for model in ['M8', 'M9', 'M10']:
        sub = multicov_df[multicov_df['model_type'] == model]
        X_sub, y_sub = prepare(sub[FEATURES], sub['abs_bias_pct'].values)
        r2_m = r2_score(y_sub, rf_combined.predict(X_sub))
        mae_m = mean_absolute_error(y_sub, rf_combined.predict(X_sub))
        print(f"  {model}: R2={r2_m:.4f}, MAE={mae_m:.2f}pp")

    # ── 6. Test diagnostic decision tree rule ──
    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC RULE TEST ON MULTI-COVARIATE DATA")
    print(f"{'=' * 60}")

    # Train a depth-3 decision tree on single-covariate data
    dt_single = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10,
                                      random_state=SEED)
    dt_single.fit(X_train[TREE_FEATURES], y_train)
    print(f"\nTwo-input tree trained on single-covariate M1/M5:")
    print(f"  R2 on single held-out (M6/M7): "
          f"{r2_score(y_test_s, dt_single.predict(X_test_s[TREE_FEATURES])):.4f}")
    rule_text = export_text(dt_single, feature_names=TREE_FEATURES,
                            show_weights=True, decimals=3)
    for line in rule_text.strip().split('\n'):
        print(f"    {line}")

    print(f"\nTree generalization to multi-covariate:")
    for model in ['M8', 'M9', 'M10']:
        sub = multicov_df[multicov_df['model_type'] == model]
        X_sub, y_sub = prepare(sub[FEATURES], sub['abs_bias_pct'].values)
        r2_m = r2_score(y_sub, dt_single.predict(X_sub[TREE_FEATURES]))
        mae_m = mean_absolute_error(y_sub, dt_single.predict(X_sub[TREE_FEATURES]))
        print(f"  {model}: R2={r2_m:.4f}, MAE={mae_m:.2f}pp")

    # ── 7. Feature importance comparison ──
    print(f"\n{'=' * 60}")
    print("FEATURE IMPORTANCE SHIFT")
    print(f"{'=' * 60}")

    # Train RF on multi-covariate only to see feature importance
    rf_multi = train_rf_single(multicov_features, multicov_target)
    imp = permutation_importance(rf_multi, multicov_features, multicov_target,
                                 n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_idx = np.argsort(imp.importances_mean)[::-1]

    print(f"\nTop-10 features (trained on multi-covariate only):")
    for rank, idx in enumerate(imp_idx[:10]):
        print(f"  {rank+1}. {FEATURES[idx]:<20} "
              f"{imp.importances_mean[idx]:.4f} ± {imp.importances_std[idx]:.4f}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    return {
        'rf_single': rf,
        'rf_combined': rf_combined,
        'multicov_df': multicov_df,
        'generalization': res_gen,
    }


if __name__ == '__main__':
    results = main()
