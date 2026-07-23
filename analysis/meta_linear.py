"""
Continuous outcome extension: test meta-model on linear Y (continuous).

Key questions:
  1. Within-regime: How well does RF predict bias in continuous-Y data?
  2. Bias decomposition: How much of binary-Y bias is non-collapsibility vs. structural?
  3. Cross-regime transfer: Does binary-Y-trained RF predict continuous-Y bias?

For continuous Y (linear DGP), non-collapsibility is absent, so any remaining
bias is purely structural (confounding/mediation/collider/selection).

Generates 7 DAGs x 2 cases x 3 lambda_xy x path params x 3 N x 20 iters
= 14,040 rows (same grid as original training data).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import time, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG

SEED = 20240625
N_ITER = 20

# Parameters (same grid as original)
LAMBDA_XY = [0.05, 0.10, 0.30]
LAMBDA_PATH = [0.10, 0.30, 0.50]
SAMPLE_SIZES = [250, 1000, 5000]
MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
CASES = ['a', 'd']

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'lpm_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

MODEL_VARY_PARAMS = {
    'M1': ['lambda_zx', 'lambda_zy'],
    'M2': ['lambda_zy'],
    'M3': ['lambda_zx'],
    'M4': ['lambda_xz'],
    'M5': ['lambda_xz', 'lambda_zy'],
    'M6': ['lambda_xz', 'lambda_yz'],
    'M7': ['lambda_yz'],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Continuous Y DGP
# ═══════════════════════════════════════════════════════════════════════════════

def generate_linear_data(model_type, case, n, params, rs=None):
    """Generate data with continuous Y (linear DGP, no logit link)."""
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
        u_z = rs.normal(0, 1, n)
        u_x = rs.normal(0, 1, n)
        u_y = rs.normal(0, 1, n)
        z_base = rs.normal(0, 1, n)
    else:  # 'd'
        u_z = rs.binomial(1, 0.5, n)
        u_x = rs.binomial(1, 0.5, n)
        u_y = rs.binomial(1, 0.5, n)
        z_base = rs.binomial(1, 0.5, n)

    # Z generation (same as dgp.py)
    if model_type in ['M1', 'M2', 'M3']:
        if case in ['a', 'c']:
            z = 0.5 * z_base + 0.5 * u_z
        else:
            z = z_base.astype(float)
    elif model_type == 'M4':
        x_init = u_x
        if case in ['a', 'c']:
            z_init = 0.5 * (lam_xz * x_init) + 0.5 * u_z
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2 * x_init - 1) + 0.5 * z_base, -100, 100)))
            z_init = rs.binomial(1, p_z).astype(float)
        z = z_init
    elif model_type == 'M5':
        x_init = u_x
        if case in ['a', 'c']:
            z_init = 0.5 * (lam_xz * x_init + u_z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2 * x_init - 1) + 0.5 * z_base, -100, 100)))
            z_init = rs.binomial(1, p_z).astype(float)
        z = z_init
    elif model_type == 'M6':
        if case in ['a', 'c']:
            z = 0.5 * (lam_xz * u_x + u_z)
        else:
            z = z_base.astype(float)
    elif model_type == 'M7':
        if case in ['a', 'c']:
            z = 0.5 * u_z + 0.5 * z_base
        else:
            z = z_base.astype(float)

    # X generation
    if model_type in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7']:
        if case in ['a', 'c']:
            x = lam_zx * z + u_x
        else:
            if case == 'b':
                z_binary = (z > np.median(z)).astype(float)
            else:
                z_binary = z.copy()
            p_x = 1 / (1 + np.exp(-np.clip(lam_zx * (2 * z_binary - 1) - 0.5, -100, 100)))
            x = rs.binomial(1, p_x).astype(float)
    else:  # M4
        x = u_x.astype(float) if case in ['c', 'd'] else u_x

    # Continuous Y generation (LINEAR — no logit, no centering, no Bernoulli draw)
    if model_type in ['M1', 'M2', 'M5']:
        y = intercept + lam_xy * x + lam_zy * z + u_y
    else:  # M3, M4, M6, M7
        y = intercept + lam_xy * x + u_y

    # Post-generation Z adjustments (same as dgp.py)
    if model_type == 'M6':
        if case in ['a', 'c']:
            z = lam_xz * x + lam_yz * y + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2*x-1) + lam_yz*(2*y-1) + 0.5*z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)
    elif model_type == 'M7':
        if case in ['a', 'c']:
            z = lam_yz * y + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_yz * (2*y-1) + 0.5*z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)
    elif model_type == 'M4':
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2*x-1) + 0.5*z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)
    elif model_type == 'M5':
        if case in ['a', 'c']:
            z = lam_xz * x + u_z
            z = z - np.mean(z)
        else:
            p_z = 1 / (1 + np.exp(-np.clip(lam_xz * (2*x-1) + 0.5*z_base, -100, 100)))
            z = rs.binomial(1, p_z).astype(float)

    return pd.DataFrame({'Y': y, 'X': x, 'Z': z})


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Meta-Features (OLS-based for continuous Y)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_linear_meta_features(data, ols_ace, unadj_ace):
    """Compute 16 meta-features for continuous Y (OLS-based)."""
    n = len(data)
    y_mean = float(data['Y'].mean())
    x_var = float(data['X'].var(ddof=1))

    # OLS Y ~ X + Z
    ols_model = sm.OLS.from_formula("Y ~ X + Z", data=data).fit()
    z_coef_mag = float(abs(ols_model.params.get('Z', 0)))
    lpm_x_coef = float(ols_model.params.get('X', 0))
    y_model_r2 = float(ols_model.rsquared)
    pred = ols_model.predict(data)
    y_pred_min = float(pred.min())
    y_pred_max = float(pred.max())

    # OLS X ~ Z
    ols_zx = sm.OLS.from_formula("X ~ Z", data=data).fit()
    z_x_coef_mag = float(abs(ols_zx.params.get('Z', 0)))
    z_x_r2 = float(ols_zx.rsquared)

    # For continuous Y, the "logit" and "LPM" estimators are the same (OLS)
    # logit_lpm_diff is defined as 0 since there's no difference
    logit_lpm_diff = 0.0

    return {
        'n': n, 'y_mean': y_mean, 'x_var': x_var,
        'logit_ace': ols_ace, 'lpm_ace': ols_ace,
        'logit_lpm_diff': logit_lpm_diff,
        'unadj_ace': unadj_ace,
        'adj_unadj_diff': abs(ols_ace - unadj_ace),
        'z_coef_mag': z_coef_mag, 'z_x_coef_mag': z_x_coef_mag,
        'z_x_r2': z_x_r2,
        'y_pred_min': y_pred_min, 'y_pred_max': y_pred_max,
        'y_pred_range': y_pred_max - y_pred_min,
        'lpm_x_coef': lpm_x_coef, 'y_model_r2': y_model_r2,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART C — Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_configs():
    """Yield (model, case, n, params_dict)."""
    for model in MODELS:
        for case in CASES:
            for lam_xy in LAMBDA_XY:
                for n in SAMPLE_SIZES:
                    vary_names = MODEL_VARY_PARAMS[model]
                    base = {
                        'lambda_zx': CONFIG.default_lambda_zx,
                        'lambda_zy': CONFIG.default_lambda_zy,
                        'lambda_xy': lam_xy,
                        'lambda_xz': CONFIG.default_lambda_xz,
                        'lambda_yz': CONFIG.default_lambda_yz,
                    }
                    if len(vary_names) == 2:
                        for v1 in LAMBDA_PATH:
                            for v2 in LAMBDA_PATH:
                                p = base.copy()
                                p[vary_names[0]] = v1
                                p[vary_names[1]] = v2
                                yield model, case, n, p
                    else:
                        for v1 in LAMBDA_PATH:
                            p = base.copy()
                            p[vary_names[0]] = v1
                            yield model, case, n, p


def generate_linear_data_full():
    """Generate the full 14,040-row continuous-Y dataset."""
    t0 = time.time()
    print("=" * 70)
    print("CONTINUOUS Y: GENERATING DATA")
    print("=" * 70)

    configs = list(_generate_configs())
    n_config = len(configs)
    n_total = n_config * N_ITER
    print(f"Configurations: {n_config}, Iterations: {N_ITER}, Total: {n_total}")

    records = []
    sim_start = time.time()

    for cfg_idx, (model, case, n, params) in enumerate(configs):
        true_ace = params['lambda_xy']  # For linear Y, ACE = lambda_xy exactly
        vary_names = MODEL_VARY_PARAMS[model]

        if (cfg_idx + 1) % 20 == 0:
            elapsed = time.time() - sim_start
            rate = (cfg_idx + 1) * N_ITER / elapsed if elapsed > 0 else 0
            print(f"  [{((cfg_idx+1)/n_config*100):.0f}%] cfg {cfg_idx+1}/{n_config} "
                  f"({rate:.0f} sim/s)")

        for it in range(N_ITER):
            rs = np.random.RandomState(SEED + cfg_idx + it * 10000)
            data = generate_linear_data(model, case, n, params, rs)

            # ACE estimates via OLS
            ols_adj = sm.OLS.from_formula("Y ~ X + Z", data=data).fit()
            ols_ace = float(ols_adj.params['X'])  # adjusted ACE = coef on X

            ols_unadj = sm.OLS.from_formula("Y ~ X", data=data).fit()
            unadj_ace = float(ols_unadj.params['X'])  # unadjusted ACE

            # Bias computation
            bias_pct = (ols_ace - true_ace) / max(abs(true_ace), 1e-10) * 100

            # Meta-features
            mf = compute_linear_meta_features(data, ols_ace, unadj_ace)
            record = {**mf,
                      'true_ace': true_ace,
                      'abs_bias_pct': abs(bias_pct),
                      'high_bias': int(abs(bias_pct) > 5),
                      'model_type': model, 'case_type': case,
                      'lambda_xy': params['lambda_xy'],
                      'vary_param_1_name': vary_names[0],
                      'vary_param_1_val': params[vary_names[0]],
                      'vary_param_2_name': vary_names[1] if len(vary_names) > 1 else 'none',
                      'vary_param_2_val': params[vary_names[1]] if len(vary_names) > 1 else 0.0}
            records.append(record)

    df = pd.DataFrame(records)
    t1 = time.time()

    print(f"\nGenerated {len(df)} rows in {t1-t0:.0f}s")
    print(f"High-bias rate: {df['high_bias'].mean()*100:.1f}%")
    print(f"Mean |bias|: {df['abs_bias_pct'].mean():.2f}%")
    print(f"Mean |bias| (binary Y comparison): 121.8%")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART D — RF Training and Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

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
    print(f"  {label:<40} R2={r2:.4f}  MAE={mae:.2f}")
    for m in sorted(per_dag.keys()):
        print(f"    {m:<6} R2={per_dag[m]['r2']:.4f}  MAE={per_dag[m]['mae']:.2f}  "
              f"hb={per_dag[m]['high_bias']:.0f}%  n={per_dag[m]['n']}")
    return {'r2': r2, 'mae': mae, 'per_dag': per_dag}


def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=1000, max_depth=10,
                               min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    return rf


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("CONTINUOUS OUTCOME EXTENSION")
    print("Decomposing structural bias vs non-collapsibility")
    print("=" * 70)

    # ── Generate continuous Y data ──
    df_lin = generate_linear_data_full()

    # Save to parquet
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../results/raw')
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, 'meta_linear_data.parquet')
    try:
        df_lin.to_parquet(parquet_path, index=False)
        print(f"\n[OK] Saved {parquet_path}")
    except Exception:
        csv_path = parquet_path.replace('.parquet', '.csv')
        df_lin.to_csv(csv_path, index=False)
        print(f"\n[OK] Saved {csv_path}")

    # ── Load original binary Y data for comparison ──
    bin_parquet = os.path.join(out_dir, 'meta_training_data.parquet')
    df_bin = pd.read_parquet(bin_parquet)
    print(f"\nLoaded binary Y data: {len(df_bin)} rows")

    # ── Analysis 1: Continuous Y — within-regime RF ──
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Continuous Y — Within-Regime RF")
    print("(train on M1/M5 linear, test on M6/M7 linear)")
    print("=" * 70)

    train_lin = df_lin[df_lin['model_type'].isin(['M1', 'M5'])]
    test_lin = df_lin[df_lin['model_type'].isin(['M6', 'M7'])]
    print(f"Train: {len(train_lin)} rows, Test: {len(test_lin)} rows")

    X_tr, y_tr = prepare(train_lin[FEATURES], train_lin['abs_bias_pct'].values)
    X_te, y_te = prepare(test_lin[FEATURES], test_lin['abs_bias_pct'].values)
    rf_lin = train_rf(X_tr, y_tr)
    eval_lin = evaluate(rf_lin, test_lin, "Continuous Y RF (held-out)")

    # Feature importance
    imp = permutation_importance(rf_lin, X_te, y_te, n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
    imp_idx = np.argsort(imp.importances_mean)[::-1]
    print(f"\n  Top-8 Feature Importances:")
    for rank, idx in enumerate(imp_idx[:8]):
        print(f"    {FEATURES[idx]:<22} {imp.importances_mean[idx]:.4f}")

    # ── Analysis 2: Bias magnitude comparison ──
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Bias Comparison — Linear vs Binary Y")
    print("=" * 70)

    for model in MODELS:
        sub_bin = df_bin[df_bin['model_type'] == model]
        sub_lin = df_lin[df_lin['model_type'] == model]
        m_bin = sub_bin['abs_bias_pct'].mean()
        m_lin = sub_lin['abs_bias_pct'].mean()
        ratio = m_lin / m_bin if m_bin > 0 else 0
        print(f"  {model:<6} binary={m_bin:>8.1f}%  linear={m_lin:>8.1f}%  "
              f"ratio={ratio:.3f}")

    overall_bin = df_bin['abs_bias_pct'].mean()
    overall_lin = df_lin['abs_bias_pct'].mean()
    print(f"  {'ALL':<6} binary={overall_bin:>8.1f}%  linear={overall_lin:>8.1f}%  "
          f"ratio={overall_lin/overall_bin:.3f}")
    print(f"  Implied non-collapsibility contribution: "
          f"{(1 - overall_lin/overall_bin)*100:.1f}% of binary-Y bias")

    # ── Analysis 3: Cross-regime transfer ──
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Cross-Regime Transfer")
    print("=" * 70)

    # 3a: Binary-trained RF → continuous Y test set
    train_bin = df_bin[df_bin['model_type'].isin(['M1', 'M5'])]
    X_tr_bin, y_tr_bin = prepare(train_bin[FEATURES],
                                  train_bin['abs_bias_pct'].values)
    rf_bin = train_rf(X_tr_bin, y_tr_bin)
    print("\n  Binary-trained RF applied to continuous Y test set:")
    eval_cross_bin2lin = evaluate(rf_bin, test_lin, "Binary→Linear transfer")

    # 3b: Continuous-trained RF → binary Y test set
    test_bin = df_bin[df_bin['model_type'].isin(['M6', 'M7'])]
    print("\n  Continuous-trained RF applied to binary Y test set:")
    eval_cross_lin2bin = evaluate(rf_lin, test_bin, "Linear→Binary transfer")

    # ── Analysis 4: Per-DAG breakdown (random split) ──
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Per-DAG Continuous Y R2 (Random 80/20 Split)")
    print("=" * 70)

    from sklearn.model_selection import train_test_split
    train_r, test_r = train_test_split(df_lin, test_size=0.2,
                                        random_state=SEED)
    Xr_tr, yr_tr = prepare(train_r[FEATURES], train_r['abs_bias_pct'].values)
    Xr_te, yr_te = prepare(test_r[FEATURES], test_r['abs_bias_pct'].values)
    rf_r = train_rf(Xr_tr, yr_tr)
    test_r['rf_pred'] = rf_r.predict(Xr_te)
    for model in sorted(test_r['model_type'].unique()):
        sub = test_r[test_r['model_type'] == model]
        r2_m = r2_score(sub['abs_bias_pct'], sub['rf_pred'])
        mae_m = mean_absolute_error(sub['abs_bias_pct'], sub['rf_pred'])
        hb = sub['high_bias'].mean() * 100
        print(f"  {model:<6} R2={r2_m:.4f}  MAE={mae_m:.2f}  hb={hb:.0f}%  "
              f"n={len(sub)}")

    # ── Analysis 5: Decision tree rule transfer ──
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Decision Tree Rule on Continuous Y")
    print("=" * 70)

    from sklearn.tree import DecisionTreeRegressor, export_text
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=SEED)
    dt.fit(X_tr, y_tr)
    r2_dt = r2_score(y_te, dt.predict(X_te))
    mae_dt = mean_absolute_error(y_te, dt.predict(X_te))
    print(f"  Decision tree (depth=3): R2={r2_dt:.4f}, MAE={mae_dt:.2f}")
    print(f"  Tree rule:")
    rule = export_text(dt, feature_names=FEATURES, show_weights=True, decimals=3)
    for line in rule.strip().split('\n'):
        print(f"    {line}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Analysis':<35} {'R2':>8} {'MAE':>8}")
    print(f"{'─'*35} {'─'*8} {'─'*8}")
    print(f"{'Binary Y (original held-out)':<35} {0.6330:>8.4f} {65.46:>7.2f}pp")
    print(f"{'Continuous Y (held-out)':<35} {eval_lin['r2']:>8.4f} {eval_lin['mae']:>7.2f}")
    print(f"{'Binary→Linear transfer':<35} {eval_cross_bin2lin['r2']:>8.4f} {eval_cross_bin2lin['mae']:>7.2f}")
    print(f"{'Linear→Binary transfer':<35} {eval_cross_lin2bin['r2']:>8.4f} {eval_cross_lin2bin['mae']:>7.2f}")
    print(f"\nBias decomposition:")
    print(f"  Mean |bias| (binary Y): {overall_bin:.1f}%")
    print(f"  Mean |bias| (linear Y): {overall_lin:.1f}%")
    print(f"  Structural bias share:  {overall_lin/overall_bin*100:.1f}%")
    print(f"  Non-collapsibility share: {(1-overall_lin/overall_bin)*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    return df_lin, rf_lin, eval_lin


if __name__ == '__main__':
    main()
