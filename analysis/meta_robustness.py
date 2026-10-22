"""
Meta-model: Robustness Analysis

Implements two remaining P1 items from the cross-validation report:
  1. P1-4: Bootstrap confidence intervals for meta-model metrics (R2, MAE, RMSE)
  2. P1-6: RF hyperparameter sensitivity analysis

Outputs: console summary and results saved to results/raw/meta_robustness_results.json
"""

import numpy as np
import pandas as pd
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 20240625

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]


def load_and_split():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, '../results/raw/meta_training_data.parquet')
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows")

    # Held-out split: train M1/M5, test M6/M7
    train = df[df['model_type'].isin(['M1', 'M5'])].copy()
    test = df[df['model_type'].isin(['M6', 'M7'])].copy()

    X_train = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(train[FEATURES].median())
    y_train = train['abs_bias_pct'].values
    X_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(test[FEATURES].median())
    y_test = test['abs_bias_pct'].values

    print(f"Train: {len(train)} rows (M1/M5)")
    print(f"Test:  {len(test)} rows (M6/M7)")
    return X_train, y_train, X_test, y_test, df


def bootstrap_ci(X_train, y_train, X_test, y_test, n_bootstrap=100):
    """Bootstrap 95% CI for R2, MAE, RMSE on held-out test set.

    Procedure: refit RF on bootstrap sample of training data, evaluate on fixed test set.
    This captures uncertainty from both Monte Carlo sampling and model training variability.
    """
    print(f"\n{'=' * 60}")
    print("P1-4: BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'=' * 60}")
    print(f"  Bootstrap iterations: {n_bootstrap}")

    r2_scores = np.empty(n_bootstrap)
    mae_scores = np.empty(n_bootstrap)
    rmse_scores = np.empty(n_bootstrap)

    rf_base = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED
    )
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    base_r2 = r2_score(y_test, y_pred_base)
    base_mae = mean_absolute_error(y_test, y_pred_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))

    n_train = X_train.shape[0]
    np.random.seed(SEED)
    t0 = time.time()
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n_train, size=n_train)
        Xb = X_train.iloc[idx]
        yb = y_train[idx]

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + i, verbose=0
        )
        rf.fit(Xb, yb)
        yp = rf.predict(X_test)

        r2_scores[i] = r2_score(y_test, yp)
        mae_scores[i] = mean_absolute_error(y_test, yp)
        rmse_scores[i] = np.sqrt(mean_squared_error(y_test, yp))

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_bootstrap}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0

    # Percentile CIs
    ci = {
        'r2': {
            'point': round(base_r2, 4),
            'ci_lower': round(np.percentile(r2_scores, 2.5), 4),
            'ci_upper': round(np.percentile(r2_scores, 97.5), 4),
            'ci_90_lower': round(np.percentile(r2_scores, 5), 4),
            'ci_90_upper': round(np.percentile(r2_scores, 95), 4),
        },
        'mae': {
            'point': round(base_mae, 2),
            'ci_lower': round(np.percentile(mae_scores, 2.5), 2),
            'ci_upper': round(np.percentile(mae_scores, 97.5), 2),
        },
        'rmse': {
            'point': round(base_rmse, 2),
            'ci_lower': round(np.percentile(rmse_scores, 2.5), 2),
            'ci_upper': round(np.percentile(rmse_scores, 97.5), 2),
        },
        'n_bootstrap': n_bootstrap,
        'method': 'Percentile bootstrap (2.5th/97.5th), refit on bootstrap training sample, evaluate on fixed test set',
    }

    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"  {'Metric':<10} {'Point Est.':>12} {'95% CI':>20}")
    print(f"  {'*'*10} {'*'*12} {'*'*20}")
    print(f"  {'R2':<10} {ci['r2']['point']:>10.4f}  "  # R2 displayed as R2 in console
          f"[{ci['r2']['ci_lower']:.4f}, {ci['r2']['ci_upper']:.4f}]")
    print(f"  {'MAE':<10} {ci['mae']['point']:>10.2f}  "
          f"[{ci['mae']['ci_lower']:.2f}, {ci['mae']['ci_upper']:.2f}]")
    print(f"  {'RMSE':<10} {ci['rmse']['point']:>10.2f}  "
          f"[{ci['rmse']['ci_lower']:.2f}, {ci['rmse']['ci_upper']:.2f}]")
    print(f"  90% CI R2: [{ci['r2']['ci_90_lower']:.4f}, {ci['r2']['ci_90_upper']:.4f}]")

    return ci


def sensitivity_analysis(X_train, y_train, X_test, y_test):
    """RF hyperparameter sensitivity: vary key parameters, report test-set R2 and MAE."""
    print(f"\n{'=' * 60}")
    print("P1-6: RF HYPERPARAMETER SENSITIVITY")
    print(f"{'=' * 60}")

    grid = {
        'n_estimators': [100, 500, 1000, 2000],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_leaf': [1, 3, 5, 10, 20],
    }

    results = []
    default_rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED
    )
    default_rf.fit(X_train, y_train)
    yp_default = default_rf.predict(X_test)
    default_r2 = r2_score(y_test, yp_default)
    default_mae = mean_absolute_error(y_test, yp_default)

    print(f"\n  Default RF (n_estimators=1000, max_depth=10, min_samples_leaf=5):")
    print(f"    R2 = {default_r2:.4f}, MAE = {default_mae:.2f}pp")

    # Vary each hyperparameter while holding others at default
    print(f"\n  Hyperparameter Sweep:")
    print(f"  {'Parameter':<25} {'Value':<8} {'R2':>8} {'MAE':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    # n_estimators
    for v in grid['n_estimators']:
        rf = RandomForestRegressor(
            n_estimators=v, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED
        )
        rf.fit(X_train, y_train)
        yp = rf.predict(X_test)
        r2 = r2_score(y_test, yp)
        mae = mean_absolute_error(y_test, yp)
        results.append({'param': f'n_est={v}', 'r2': round(r2, 4), 'mae': round(mae, 2)})
        flag = ' ← default' if v == 1000 else ''
        print(f"  {'n_estimators':<25} {v:<8} {r2:>8.4f} {mae:>8.2f}{flag}")

    # max_depth
    for v in grid['max_depth']:
        rf = RandomForestRegressor(
            n_estimators=1000, max_depth=v, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED
        )
        rf.fit(X_train, y_train)
        yp = rf.predict(X_test)
        r2 = r2_score(y_test, yp)
        mae = mean_absolute_error(y_test, yp)
        label = str(v) if v is not None else 'None'
        results.append({'param': f'depth={label}', 'r2': round(r2, 4), 'mae': round(mae, 2)})
        flag = ' ← default' if v == 10 else ''
        print(f"  {'max_depth':<25} {label:<8} {r2:>8.4f} {mae:>8.2f}{flag}")

    # min_samples_leaf
    for v in grid['min_samples_leaf']:
        rf = RandomForestRegressor(
            n_estimators=1000, max_depth=10, min_samples_leaf=v,
            n_jobs=-1, random_state=SEED
        )
        rf.fit(X_train, y_train)
        yp = rf.predict(X_test)
        r2 = r2_score(y_test, yp)
        mae = mean_absolute_error(y_test, yp)
        results.append({'param': f'leaf={v}', 'r2': round(r2, 4), 'mae': round(mae, 2)})
        flag = ' ← default' if v == 5 else ''
        print(f"  {'min_samples_leaf':<25} {v:<8} {r2:>8.4f} {mae:>8.2f}{flag}")

    # Stability across random seeds
    print(f"  Seed Stability (default RF, 10 seeds):")
    seed_results = []
    for seed_offset in range(10):
        rf = RandomForestRegressor(
            n_estimators=1000, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + seed_offset
        )
        rf.fit(X_train, y_train)
        yp = rf.predict(X_test)
        r2 = r2_score(y_test, yp)
        mae = mean_absolute_error(y_test, yp)
        seed_results.append({'r2': r2, 'mae': mae})

    seed_r2 = [s['r2'] for s in seed_results]
    seed_mae = [s['mae'] for s in seed_results]
    print(f"  R2  range: [{np.min(seed_r2):.4f}, {np.max(seed_r2):.4f}], "
          f"mean {np.mean(seed_r2):.4f}, SD {np.std(seed_r2):.4f}")
    print(f"  MAE range: [{np.min(seed_mae):.2f}, {np.max(seed_mae):.2f}], "
          f"mean {np.mean(seed_mae):.2f}, SD {np.std(seed_mae):.4f}")

    return {
        'default': {'r2': round(default_r2, 4), 'mae': round(default_mae, 2)},
        'sweep': results,
        'seed_stability': {
            'r2_mean': round(np.mean(seed_r2), 4),
            'r2_sd': round(np.std(seed_r2), 4),
            'r2_range': [round(np.min(seed_r2), 4), round(np.max(seed_r2), 4)],
            'mae_mean': round(np.mean(seed_mae), 2),
            'mae_sd': round(np.std(seed_mae), 4),
        }
    }


if __name__ == '__main__':
    t0 = time.time()
    X_train, y_train, X_test, y_test, df = load_and_split()

    # P1-4: Bootstrap CIs
    ci = bootstrap_ci(X_train, y_train, X_test, y_test, n_bootstrap=100)

    # P1-6: Sensitivity analysis
    sa = sensitivity_analysis(X_train, y_train, X_test, y_test)

    # Save results
    results = {'bootstrap_ci': ci, 'sensitivity_analysis': sa}
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/raw')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'meta_robustness_results.json')
    with open(out_path, 'w') as f:
        json.dumps(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
