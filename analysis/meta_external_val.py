"""
External validation: test meta-model on data from structurally different DGPs.

Conditions:
  1. Probit Y:         Y generated with probit link (Φ) instead of logit
  2. Extended range:   λ_xy / λ_path / N values outside training range
  3. Uniform Z:        Z ~ Uniform(-2,2) instead of Normal(0,1)

Evaluates whether the RF trained on logit-link Normal-Z data generalizes.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import time, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

SEED = 20240625
FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

# ── Link helpers ────────────────────────────────────────────────────────────

def logit_link(x):
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

def probit_link(x):
    return norm.cdf(x)


# ═══════════════════════════════════════════════════════════════════════════
# PART A — Parameterized DGP
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(model_type, case, n, params, link_fn, z_dist='normal', rs=None):
    """Generate data from a corrected DAG with parameterized link and Z distribution."""
    if rs is None:
        rs = np.random.RandomState(SEED)

    lam_zx = params.get('lambda_zx', 0.3)
    lam_zy = params.get('lambda_zy', 0.3)
    lam_xy = params.get('lambda_xy', 0.1)
    lam_xz = params.get('lambda_xz', 0.3)
    lam_yz = params.get('lambda_yz', 0.3)
    intercept = params.get('intercept', 0.0)

    # Disturbances
    if case in ['a', 'c']:
        if z_dist == 'uniform':
            u_z = rs.uniform(-2, 2, n)
            z_base = rs.uniform(-2, 2, n)
        else:
            u_z = rs.normal(0, 1, n)
            z_base = rs.normal(0, 1, n)
        u_x = rs.normal(0, 1, n)
        u_y = rs.normal(0, 1, n)
    else:
        u_z = rs.binomial(1, 0.5, n)
        z_base = rs.binomial(1, 0.5, n)
        u_x = rs.binomial(1, 0.5, n)
        u_y = rs.binomial(1, 0.5, n)

    # Z for M1/M2/M3 (exogenous)
    if model_type in ['M1', 'M2', 'M3']:
        if z_dist == 'uniform' and case in ['a', 'c']:
            z = u_z
        elif case in ['a', 'c']:
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
            p_x = 1 / (1 + np.exp(-np.clip(lam_zx * (2 * z_binary - 1) - 0.5, -100, 100)))
            x = rs.binomial(1, p_x).astype(float)
    else:
        x = u_x.astype(float) if case in ['c', 'd'] else u_x

    # Z for M4/M5 (after X, before Y)
    if model_type in ['M4', 'M5']:
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2 * x - 1) + 0.5 * z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)

    # Y
    if model_type in ['M1', 'M2', 'M5']:
        y_eta = intercept + lam_xy * x + lam_zy * z + u_y
    else:
        y_eta = intercept + lam_xy * x + u_y
    y_eta = y_eta - np.mean(y_eta)
    y_prob = link_fn(y_eta)
    y = rs.binomial(1, y_prob).astype(float)

    # Z for M6/M7 (after Y)
    if model_type == 'M6':
        if case in ['a', 'c']:
            z = lam_xz * x + lam_yz * y + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2 * x - 1) + lam_yz * (2 * y - 1) + 0.5 * z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)
    elif model_type == 'M7':
        if case in ['a', 'c']:
            z = lam_yz * y + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_yz * (2 * y - 1) + 0.5 * z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)

    return pd.DataFrame({'Y': y, 'X': x, 'Z': z})


def true_ace_generic(model_type, params, link_fn, z_dist='normal',
                     case='a', n_mc=200000, seed=42):
    """Monte Carlo total effect for any link function, Z distribution, and case.

    Case a uses continuous normal (or uniform) variables; case d uses binary
    X/Z and Bernoulli errors. The centering constant must match the case, or
    the target estimand is shifted.
    """
    rs = np.random.RandomState(seed)
    lam_zx = params.get('lambda_zx', 0.3)
    lam_zy = params.get('lambda_zy', 0.3)
    lam_xy = params.get('lambda_xy', 0.1)
    lam_xz = params.get('lambda_xz', 0.3)
    lam_yz = params.get('lambda_yz', 0.3)
    intercept = params.get('intercept', 0.0)

    if case == 'a':
        if z_dist == 'uniform':
            u_z = rs.uniform(-2, 2, n_mc)
            z_base = rs.uniform(-2, 2, n_mc)
        else:
            u_z = rs.normal(0, 1, n_mc)
            z_base = rs.normal(0, 1, n_mc)
        u_x = rs.normal(0, 1, n_mc)
        u_y = rs.normal(0, 1, n_mc)

        # Z for M1/M2/M3
        if model_type in ['M1', 'M2', 'M3']:
            if z_dist == 'uniform':
                z = u_z
            else:
                z = 0.5 * z_base + 0.5 * u_z

        # X
        if model_type in ['M1', 'M3']:
            x = lam_zx * z + u_x
        else:
            x = u_x

        # Z for M4/M5
        if model_type in ['M4', 'M5']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)

        # Observed Y and centering
        if model_type in ['M1', 'M2', 'M5']:
            y_eta = intercept + lam_xy * x + lam_zy * z + u_y
        else:
            y_eta = intercept + lam_xy * x + u_y
        centering = np.mean(y_eta)

        # Potential outcomes (total effect)
        if model_type == 'M5':
            z1 = lam_xz * 1 + u_z
            z0 = lam_xz * 0 + u_z
            eta1 = intercept + lam_xy * 1 + lam_zy * z1 + u_y - centering
            eta0 = intercept + lam_xy * 0 + lam_zy * z0 + u_y - centering
        elif model_type in ['M1', 'M2']:
            eta1 = intercept + lam_xy * 1 + lam_zy * z + u_y - centering
            eta0 = intercept + lam_xy * 0 + lam_zy * z + u_y - centering
        else:
            eta1 = intercept + lam_xy * 1 + u_y - centering
            eta0 = intercept + lam_xy * 0 + u_y - centering
    elif case == 'd':
        u_z = rs.binomial(1, 0.5, n_mc)
        u_x = rs.binomial(1, 0.5, n_mc)
        u_y = rs.binomial(1, 0.5, n_mc)
        z_base = rs.binomial(1, 0.5, n_mc)

        # Z for M1/M2/M3
        if model_type in ['M1', 'M2', 'M3']:
            z = z_base.astype(float)

        # X
        if model_type in ['M1', 'M3']:
            p_x = logit_link(lam_zx * (2 * z - 1) - 0.5)
            x = rs.binomial(1, p_x, n_mc).astype(float)
        else:
            x = u_x.astype(float)

        # Z for M4/M5
        if model_type in ['M4', 'M5']:
            p_z = logit_link(lam_xz * (2 * x - 1) + 0.5 * z_base)
            z = rs.binomial(1, p_z, n_mc).astype(float)

        # Observed Y and centering
        if model_type in ['M1', 'M2', 'M5']:
            y_eta = intercept + lam_xy * x + lam_zy * z + u_y
        else:
            y_eta = intercept + lam_xy * x + u_y
        centering = np.mean(y_eta)

        # Potential outcomes (total effect)
        if model_type == 'M5':
            p_z1 = logit_link(lam_xz * (2 * 1 - 1) + 0.5 * z_base)
            p_z0 = logit_link(lam_xz * (2 * 0 - 1) + 0.5 * z_base)
            z1 = rs.binomial(1, p_z1, n_mc).astype(float)
            z0 = rs.binomial(1, p_z0, n_mc).astype(float)
            eta1 = intercept + lam_xy * 1 + lam_zy * z1 + u_y - centering
            eta0 = intercept + lam_xy * 0 + lam_zy * z0 + u_y - centering
        elif model_type in ['M1', 'M2']:
            eta1 = intercept + lam_xy * 1 + lam_zy * z + u_y - centering
            eta0 = intercept + lam_xy * 0 + lam_zy * z + u_y - centering
        else:
            eta1 = intercept + lam_xy * 1 + u_y - centering
            eta0 = intercept + lam_xy * 0 + u_y - centering
    else:
        raise ValueError(f"unsupported case: {case}")

    return float(np.mean(link_fn(eta1) - link_fn(eta0)))


def compute_meta_features(data, logit_ace_val, lpm_ace_val, unadj_ace_val):
    """Compute 16 meta-features (same formulas as meta_generate.py)."""
    n = len(data)
    y_mean = float(data['Y'].mean())
    x_var = float(data['X'].var(ddof=1))

    logit_model = sm.Logit.from_formula("Y ~ X + Z", data=data).fit(disp=False, maxiter=100)
    z_coef_mag = float(abs(logit_model.params.get('Z', 0)))

    pred = logit_model.predict(data)
    y_pred_min = float(pred.min())
    y_pred_max = float(pred.max())

    ols_zx = sm.OLS.from_formula("X ~ Z", data=data).fit()
    z_x_coef_mag = float(abs(ols_zx.params.get('Z', 0)))
    z_x_r2 = float(ols_zx.rsquared)

    # Use HC3 for LPM (same as original)
    lpm_model = sm.OLS.from_formula("Y ~ X + Z", data=data).fit(cov_type='HC3')
    lpm_x_coef = float(lpm_model.params.get('X', 0))

    ols_y = sm.OLS.from_formula("Y ~ X + Z", data=data).fit()
    y_model_r2 = float(ols_y.rsquared)

    return {
        'n': n, 'y_mean': y_mean, 'x_var': x_var,
        'logit_ace': logit_ace_val, 'lpm_ace': lpm_ace_val,
        'logit_lpm_diff': abs(logit_ace_val - lpm_ace_val),
        'unadj_ace': unadj_ace_val,
        'adj_unadj_diff': abs(logit_ace_val - unadj_ace_val),
        'z_coef_mag': z_coef_mag, 'z_x_coef_mag': z_x_coef_mag,
        'z_x_r2': z_x_r2,
        'y_pred_min': y_pred_min, 'y_pred_max': y_pred_max,
        'y_pred_range': y_pred_max - y_pred_min,
        'lpm_x_coef': lpm_x_coef, 'y_model_r2': y_model_r2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART D — External Validation Conditions
# ═══════════════════════════════════════════════════════════════════════════

def _full_param_grid(models, cases, lambdas_xy, path_vals, sizes):
    """Generate configs for DAGs with varying path parameters."""
    configs = []
    path_params_by_model = {
        'M1': ['lambda_zx', 'lambda_zy'],
        'M2': ['lambda_zy'],
        'M3': ['lambda_zx'],
        'M4': ['lambda_xz'],
        'M5': ['lambda_xz', 'lambda_zy'],
        'M6': ['lambda_xz', 'lambda_yz'],
        'M7': ['lambda_yz'],
    }
    for model in models:
        pp = path_params_by_model[model]
        for case in cases:
            for lam_xy in lambdas_xy:
                for n in sizes:
                    # Generate path param combinations
                    if len(pp) == 2:
                        for v1 in path_vals:
                            for v2 in path_vals:
                                params = {'lambda_xy': lam_xy, pp[0]: v1, pp[1]: v2}
                                configs.append((model, case, n, params))
                    else:
                        for v in path_vals:
                            params = {'lambda_xy': lam_xy, pp[0]: v}
                            configs.append((model, case, n, params))
    return configs


def define_conditions():
    """Return list of (condition_name, configs, link_fn, z_dist, desc)."""
    models_4 = ['M1', 'M5', 'M6', 'M7']
    models_3 = ['M1', 'M6', 'M7']
    cases_2 = ['a', 'd']
    std_path = [0.10, 0.30, 0.50]
    std_lam_xy = [0.05, 0.10, 0.30]
    std_n = [250, 1000, 5000]

    return [
        (
            '1_Probit',
            _full_param_grid(models_4, cases_2, std_lam_xy, std_path, std_n),
            probit_link, 'normal',
            'Probit Y, original parameter ranges'
        ),
        (
            '2_ExtendedRange',
            _full_param_grid(models_3, cases_2, [0.01, 0.50, 1.0], [0.05, 0.70], [100, 10000]),
            logit_link, 'normal',
            'Logit Y, extended λ [0.01-1.0], N [100-10000]'
        ),
        (
            '3_UniformZ',
            _full_param_grid(models_4, cases_2, std_lam_xy, std_path, [1000]),
            logit_link, 'uniform',
            'Logit Y, Z~Uniform(-2,2), N=1000'
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# PART E — Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(name, configs, link_fn, z_dist):
    """Run all simulations for one external validation condition."""
    n_config = len(configs)
    total = n_config * 20
    records = []
    sim_start = time.time()

    print(f"\n{'─' * 55}")
    print(f"Condition: {name} ({len(configs)} configs x 20 = {total} sims)")
    print(f"{'─' * 55}")

    # Precompute true ACE
    ace_cache = {}
    for model, case, n, params in configs:
        key = (model, case, tuple(sorted(params.items())))
        if key not in ace_cache:
            ace_cache[key] = true_ace_generic(model, params, link_fn, z_dist,
                                             case=case)
    print(f"  True ACE: {len(ace_cache)} unique configs cached")

    for cfg_idx, (model, case, n, params) in enumerate(configs):
        if (cfg_idx + 1) % 15 == 0:
            elapsed = time.time() - sim_start
            rate = (cfg_idx + 1) * 20 / elapsed if elapsed > 0 else 0
            print(f"  [{((cfg_idx+1)/n_config*100):.0f}%] cfg {cfg_idx+1}/{n_config} "
                  f"({rate:.0f} sim/s)")

        key = (model, case, tuple(sorted(params.items())))
        true_ace = ace_cache[key]

        for it in range(20):
            seed = SEED + cfg_idx + it * 10000
            rs = np.random.RandomState(seed)
            data = generate_data(model, case, n, params, link_fn, z_dist, rs)

            # ACE estimates (always logit g-comp + LPM, same as training)
            logit_mod = sm.Logit.from_formula("Y ~ X + Z", data=data).fit(disp=False, maxiter=100)
            df1, df0 = data.copy(), data.copy()
            df1['X'] = 1.0; df0['X'] = 0.0
            logit_ace = float(np.mean(logit_mod.predict(df1) - logit_mod.predict(df0)))

            lpm_mod = sm.OLS.from_formula("Y ~ X + Z", data=data).fit(cov_type='HC3')
            lpm_ace = float(np.mean(lpm_mod.predict(df1) - lpm_mod.predict(df0)))

            unadj_mod = sm.Logit.from_formula("Y ~ X", data=data).fit(disp=False, maxiter=100)
            unadj_ace = float(np.mean(unadj_mod.predict(df1) - unadj_mod.predict(df0)))

            # Meta-features
            mf = compute_meta_features(data, logit_ace, lpm_ace, unadj_ace)
            bias_pct = (logit_ace - true_ace) / abs(true_ace) * 100

            record = {**mf, 'true_ace': true_ace,
                      'abs_bias_pct': abs(bias_pct),
                      'high_bias': int(abs(bias_pct) > 5),
                      'model_type': model, 'case_type': case,
                      'condition': name,
                      'n_config': n, 'lambda_xy': params.get('lambda_xy')}
            records.append(record)

    elapsed = time.time() - sim_start
    print(f"  Done: {len(records)} recs in {elapsed:.0f}s")
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# PART F — Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def prepare(X_df, y):
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def evaluate(rf, df, label):
    X, y = prepare(df[FEATURES], df['abs_bias_pct'].values)
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    per_dag = {}
    df = df.copy()
    df['pred'] = y_pred
    for m in sorted(df['model_type'].unique()):
        sub = df[df['model_type'] == m]
        per_dag[m] = {
            'r2': r2_score(sub['abs_bias_pct'], sub['pred']),
            'mae': mean_absolute_error(sub['abs_bias_pct'], sub['pred']),
            'n': len(sub),
            'high_bias': sub['high_bias'].mean() * 100,
        }

    print(f"  {label:<35} R2={r2:.4f}  MAE={mae:.2f}pp")
    for m in sorted(per_dag.keys()):
        print(f"    {m:<6} R2={per_dag[m]['r2']:.4f}  MAE={per_dag[m]['mae']:.2f}pp  "
              f"hb={per_dag[m]['high_bias']:.0f}%  n={per_dag[m]['n']}")
    return {'r2': r2, 'mae': mae, 'per_dag': per_dag}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("EXTERNAL VALIDATION")
    print("=" * 60)

    # ── 1. Train RF on original logit data ──
    base = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base, '../results/raw/meta_training_data.parquet')
    df = pd.read_parquet(parquet_path)

    train = df[df['model_type'].isin(['M1', 'M5'])]
    X_train, y_train = prepare(train[FEATURES], train['abs_bias_pct'].values)
    rf = RandomForestRegressor(n_estimators=1000, max_depth=10,
                                min_samples_leaf=5, n_jobs=2, random_state=SEED)
    rf.fit(X_train, y_train)
    print(f"\nRF trained on M1/M5 ({len(train)} rows)")

    # Original held-out baselines
    test_orig = df[df['model_type'].isin(['M6', 'M7'])]
    X_to, y_to = prepare(test_orig[FEATURES], test_orig['abs_bias_pct'].values)
    r2_orig = r2_score(y_to, rf.predict(X_to))
    mae_orig = mean_absolute_error(y_to, rf.predict(X_to))
    print(f"Original held-out (M6/M7): R2={r2_orig:.4f}  MAE={mae_orig:.2f}pp")

    # ── 2. Generate & evaluate external validation data ──
    conditions = define_conditions()
    results = {}

    for name, configs, link_fn, z_dist, desc in conditions:
        print(f"\n{'=' * 60}")
        print(f"CONDITION: {name} — {desc}")
        print(f"{'=' * 60}")
        ev_df = run_condition(name, configs, link_fn, z_dist)
        print(f"\n  Summary: n={len(ev_df)}, high-bias={ev_df['high_bias'].mean()*100:.1f}%, "
              f"mean |bias|={ev_df['abs_bias_pct'].mean():.1f}%")

        results[name] = {
            'df': ev_df,
            'eval': evaluate(rf, ev_df, f"RF applied to {name}"),
        }

    # ── 2b. Save external validation data for bootstrap ──
    out_dir = os.path.join(base, '../results/raw')
    os.makedirs(out_dir, exist_ok=True)
    for name in results:
        path = os.path.join(out_dir, f'meta_ext_{name}.parquet')
        results[name]['df'].to_parquet(path, index=False)
        print(f"  Saved {path} ({len(results[name]['df'])} rows)")

    # ── 3. Combined evaluation ──
    print(f"\n{'=' * 60}")
    print("COMBINED: Train on original + external, test on original held-out")
    print(f"{'=' * 60}")

    all_ext = pd.concat([r['df'] for r in results.values()], ignore_index=True)
    X_comb = pd.concat([X_train, prepare(all_ext[FEATURES],
                                          all_ext['abs_bias_pct'].values)[0]],
                        ignore_index=True)
    y_comb = np.concatenate([y_train, all_ext['abs_bias_pct'].values])
    rf_comb = RandomForestRegressor(n_estimators=1000, max_depth=10,
                                     min_samples_leaf=5, n_jobs=2, random_state=SEED)
    rf_comb.fit(X_comb, y_comb)
    r2_comb = r2_score(y_to, rf_comb.predict(X_to))
    mae_comb = mean_absolute_error(y_to, rf_comb.predict(X_to))
    print(f"  Original held-out (M6/M7): R2={r2_comb:.4f}  MAE={mae_comb:.2f}pp")
    print(f"  (vs baseline R2={r2_orig:.4f}, MAE={mae_orig:.2f}pp)")

    for name in results:
        ev_df = results[name]['df']
        X_ev, y_ev = prepare(ev_df[FEATURES], ev_df['abs_bias_pct'].values)
        r2_m = r2_score(y_ev, rf_comb.predict(X_ev))
        print(f"  {name:<20} R2={r2_m:.4f}")

    # ── 4. Probit → retrain → re-test (if probit performance dropped) ──
    probit_r2 = results.get('1_Probit', {}).get('eval', {}).get('r2', 0)
    if probit_r2 < 0.50:
        print(f"\n{'=' * 60}")
        print("PROBIT REMEDIATION: Retrain on probit data")
        print(f"{'=' * 60}")
        probit_df = results['1_Probit']['df']
        X_prob, y_prob = prepare(probit_df[FEATURES], probit_df['abs_bias_pct'].values)
        rf_prob = RandomForestRegressor(n_estimators=1000, max_depth=10,
                                         min_samples_leaf=5, n_jobs=2, random_state=SEED)
        rf_prob.fit(X_prob, y_prob)
        r2_prob_test = r2_score(y_to, rf_prob.predict(X_to))
        print(f"  Retrained on probit → test on original held-out: R2={r2_prob_test:.4f}")

    # ── 5. Summary table ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Condition':<22} {'R2':>8} {'MAE':>8} {'N_sims':>8}")
    print(f"{'─'*22} {'─'*8} {'─'*8} {'─'*8}")
    print(f"{'Original held-out':<22} {r2_orig:>8.4f} {mae_orig:>7.2f}pp {'':>8}")
    for name in results:
        e = results[name]['eval']
        n_df = len(results[name]['df'])
        print(f"{name:<22} {e['r2']:>8.4f} {e['mae']:>7.2f}pp {n_df:>8}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    return results


if __name__ == '__main__':
    main()
