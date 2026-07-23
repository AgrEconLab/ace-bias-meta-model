"""
Meta-model: Training Data Generation

Generates a diverse dataset of simulated scenarios with systematic parameter
variation to train a meta-model that predicts ACE bias from observable features.

Parameter grid: 7 models × 2 cases × 3 λ_xy × path_params × 3 N = 702 configs
Each config: 20 MC iterations → 14,040 total rows.
"""

import numpy as np
import pandas as pd
import sys, os, time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgp import simulate_model, get_true_ace_case
from estimators import logit_ace_gcomp, lpm_ace_ols
from config import CONFIG, DEFAULT_SEED

warnings.filterwarnings('ignore')

SEED = DEFAULT_SEED
N_ITER = 20

# Parameter grid
LAMBDA_XY = [0.05, 0.10, 0.30]
LAMBDA_PATH = [0.10, 0.30, 0.50]  # path strengths
SAMPLE_SIZES = [250, 1000, 5000]
MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
CASES = ['a', 'd']

# Model-specific varying parameters (beyond lambda_xy)
MODEL_VARY_PARAMS = {
    'M1': ['lambda_zx', 'lambda_zy'],  # confounding
    'M2': ['lambda_zy'],               # outcome predictor (Z->Y only)
    'M3': ['lambda_zx'],               # exposure predictor (Z->X only)
    'M4': ['lambda_xz'],               # exposure descendant (X->Z, overcontrol)
    'M5': ['lambda_xz', 'lambda_zy'],  # mediation
    'M6': ['lambda_xz', 'lambda_yz'],  # collider
    'M7': ['lambda_yz'],               # outcome descendant
}


def _generate_configs():
    """Yield (model, case, n, params_dict) for each configuration."""
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


def _params_key(params):
    """Deterministic string key for true ACE cache."""
    return '|'.join(f"{k}={v:.4f}" for k, v in sorted(params.items()))


def generate_training_data():
    t0 = time.time()
    print("=" * 70)
    print("META-MODEL: GENERATING TRAINING DATA")
    print("=" * 70)

    configs = list(_generate_configs())
    n_config = len(configs)
    n_total = n_config * N_ITER
    print(f"\nConfigurations: {n_config}")
    print(f"Iterations per config: {N_ITER}")
    print(f"Total simulations: {n_total}")
    print(f"Parameter grid: {len(MODELS)} models × {len(CASES)} cases "
          f"× {len(LAMBDA_XY)} λ_xy × {len(LAMBDA_PATH)} path × {len(SAMPLE_SIZES)} N")

    # ── Precompute true ACE for each unique config ──
    print("\nPrecomputing true ACE values...")
    true_ace_cache = {}
    for idx, (model, case, n, params) in enumerate(configs):
        key = _params_key(params) + f"|model={model}|case={case}"
        if key not in true_ace_cache:
            if (idx + 1) % 50 == 0:
                print(f"  true ACE: {idx+1}/{n_config}")
            true_ace_cache[key] = get_true_ace_case(
                model, case, params, n_mc=100000, seed=42
            )

    # ── Run simulations ──
    print(f"\nRunning {n_config} × {N_ITER} = {n_total} simulations...")
    records = []
    sim_start = time.time()

    for cfg_idx, (model, case, n, params) in enumerate(configs):
        key = _params_key(params) + f"|model={model}|case={case}"
        true_ace = true_ace_cache[key]
        vary_names = MODEL_VARY_PARAMS[model]

        if (cfg_idx + 1) % 20 == 0:
            elapsed = time.time() - sim_start
            rate = (cfg_idx + 1) * N_ITER / elapsed if elapsed > 0 else 0
            pct = (cfg_idx + 1) / n_config * 100
            print(f"  [{pct:.0f}%] config {cfg_idx+1}/{n_config} "
                  f"({rate:.0f} sim/s, {elapsed:.0f}s elapsed)")

        for it in range(N_ITER):
            seed = SEED + cfg_idx + it * 10000
            data = simulate_model(model, case, n=n, params=params, seed=seed)

            # Estimate ACE: adjusted logit, LPM, unadjusted logit
            logit_res = logit_ace_gcomp(data, bootstrap=False)
            lpm_res = lpm_ace_ols(data, bootstrap=False)
            unadj_res = logit_ace_gcomp(data, z_name=None, bootstrap=False)

            logit_ace = logit_res['ace']
            lpm_ace = lpm_res['ace']
            unadj_ace = unadj_res['ace']

            # ── Meta-features ──
            y_mean = float(data['Y'].mean())
            x_var = float(data['X'].var(ddof=1))

            # Logit model Z coefficient (magnitude)
            import statsmodels.api as sm
            logit_model = sm.Logit.from_formula("Y ~ X + Z", data=data).fit(
                disp=False, maxiter=100)
            z_coef_mag = float(abs(logit_model.params.get('Z', 0)))
            z_coef_pval = float(logit_model.pvalues.get('Z', 1.0))

            # OLS X~Z for Z→X association
            ols_zx = sm.OLS.from_formula("X ~ Z", data=data).fit()
            z_x_coef_mag = float(abs(ols_zx.params.get('Z', 0)))
            z_x_pval = float(ols_zx.pvalues.get('Z', 1.0))
            z_x_r2 = float(ols_zx.rsquared)

            # Predicted probability range from logit model
            pred = logit_model.predict(data)
            y_pred_min = float(pred.min())
            y_pred_max = float(pred.max())

            # LPM coefficient on X (alternative ACE approximation)
            lpm_model = sm.OLS.from_formula("Y ~ X + Z", data=data).fit(cov_type='HC3')
            lpm_x_coef = float(lpm_model.params.get('X', 0))

            # Z model R² from the OLS counterpart
            ols_y = sm.OLS.from_formula("Y ~ X + Z", data=data).fit()
            y_model_r2 = float(ols_y.rsquared)

            # ── Bias computation ──
            logit_bias_pct = (logit_ace - true_ace) / abs(true_ace) * 100
            lpm_bias_pct = (lpm_ace - true_ace) / abs(true_ace) * 100

            record = {
                # Meta-features (observable in real data)
                'n': n,
                'y_mean': y_mean,
                'x_var': x_var,
                'logit_ace': logit_ace,
                'logit_se': logit_res.get('se', np.nan),
                'lpm_ace': lpm_ace,
                'lpm_se': lpm_res.get('se', np.nan),
                'logit_lpm_diff': abs(logit_ace - lpm_ace),
                'unadj_ace': unadj_ace,
                'adj_unadj_diff': abs(logit_ace - unadj_ace),
                'z_coef_mag': z_coef_mag,
                'z_coef_pval': z_coef_pval,
                'z_x_coef_mag': z_x_coef_mag,
                'z_x_pval': z_x_pval,
                'z_x_r2': z_x_r2,
                'y_pred_min': y_pred_min,
                'y_pred_max': y_pred_max,
                'y_pred_range': y_pred_max - y_pred_min,
                'lpm_x_coef': lpm_x_coef,
                'y_model_r2': y_model_r2,

                # Targets
                'true_ace': true_ace,
                'logit_bias_pct': logit_bias_pct,
                'lpm_bias_pct': lpm_bias_pct,
                'abs_bias_pct': abs(logit_bias_pct),
                'abs_bias_lpm_pct': abs(lpm_bias_pct),
                'high_bias': int(abs(logit_bias_pct) > 5),
                'lpm_wins': int(abs(lpm_bias_pct) < abs(logit_bias_pct)),

                # Metadata (for validation only, not features)
                'model_type': model,
                'case_type': case,
                'lambda_xy': params['lambda_xy'],
                'vary_param_1_name': vary_names[0],
                'vary_param_1_val': params[vary_names[0]],
            }

            if len(vary_names) > 1:
                record['vary_param_2_name'] = vary_names[1]
                record['vary_param_2_val'] = params[vary_names[1]]
            else:
                record['vary_param_2_name'] = 'none'
                record['vary_param_2_val'] = 0.0

            records.append(record)

    df = pd.DataFrame(records)
    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"GENERATION COMPLETE")
    print(f"  Total rows: {len(df)}")
    print(f"  Configurations: {n_config}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / n_total:.4f}s per sim)")
    print(f"  High-bias scenarios: {df['high_bias'].mean()*100:.1f}%")
    print(f"  LPM-wins scenarios: {df['lpm_wins'].mean()*100:.1f}%")
    print(f"  Mean |bias|: {df['abs_bias_pct'].mean():.1f}%")
    print(f"  Mean |bias| (LPM): {df['abs_bias_lpm_pct'].mean():.1f}%")

    return df


if __name__ == '__main__':
    df = generate_training_data()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../results/raw')
    os.makedirs(out_dir, exist_ok=True)

    # Save as Parquet (smaller, faster) with CSV fallback
    parquet_path = os.path.join(out_dir, 'meta_training_data.parquet')
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"\n[OK] Saved {parquet_path} ({os.path.getsize(parquet_path) / 1024:.0f} KB)")
    except Exception:
        csv_path = parquet_path.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Saved {csv_path} ({os.path.getsize(csv_path) / 1024:.0f} KB)")

    # Quick feature summary
    print(f"\n{'=' * 70}")
    print("FEATURE SUMMARY")
    print(f"{'=' * 70}")
    meta_cols = ['n', 'y_mean', 'x_var', 'logit_lpm_diff', 'adj_unadj_diff',
                 'z_coef_mag', 'z_x_coef_mag', 'y_pred_range', 'y_model_r2']
    print(df[meta_cols].describe().to_string())
