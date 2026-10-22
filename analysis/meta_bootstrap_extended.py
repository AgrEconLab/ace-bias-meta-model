"""
Extended Bootstrap CI: binary Y, linear Y, and external validation.

For each bootstrap iteration (100 iters):
  1. Resample M1/M5 from binary-Y training data with replacement
  2. Refit RF (200 trees, depth=10, leaf=5)
  3. Evaluate on:
     a. Binary Y held-out (M6/M7 from meta_training_data.parquet)
     b. Linear Y held-out (M6/M7 from meta_linear_data.parquet)
     c. External validation conditions (meta_ext_*.parquet, if available)

Outputs: results/raw/meta_bootstrap_extended_results.json
"""

import numpy as np
import pandas as pd
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 20240625
N_BOOTSTRAP = 100

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

TARGET = 'abs_bias_pct'

# One block per (model, case, n, lambda_xy, path parameters); the 20 Monte
# Carlo iterations within a block are not independent draws.
GROUP_COLS = [
    'model_type', 'case_type', 'n', 'lambda_xy',
    'vary_param_1_name', 'vary_param_1_val',
    'vary_param_2_name', 'vary_param_2_val',
]

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, '../results/raw')


def prepare(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median()
    return X.fillna(med), y


def load_binary():
    """Load binary-Y training (M1/M5) and held-out test (M6/M7)."""
    path = os.path.join(RAW_DIR, 'meta_training_data.parquet')
    df = pd.read_parquet(path)
    train = df[df['model_type'].isin(['M1', 'M5'])].copy()
    test = df[df['model_type'].isin(['M6', 'M7'])].copy()
    print(f"  Binary Y: train={len(train)}, test={len(test)}")
    return train, test


def load_continuous():
    """Load continuous-Y training (M1/M5) and held-out test (M6/M7)."""
    path = os.path.join(RAW_DIR, 'meta_linear_data.parquet')
    df = pd.read_parquet(path)
    train = df[df['model_type'].isin(['M1', 'M5'])].copy()
    test = df[df['model_type'].isin(['M6', 'M7'])].copy()
    print(f"  Linear Y: train={len(train)}, test={len(test)}")
    return train, test


def load_external_conditions():
    """Load external validation parquet files if they exist."""
    conditions = {}
    for cond_name in ['1_Probit', '2_ExtendedRange', '3_UniformZ']:
        path = os.path.join(RAW_DIR, f'meta_ext_{cond_name}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            X_te, y_te = prepare(df[FEATURES], df[TARGET].values)
            conditions[cond_name] = (X_te, y_te, len(df))
            print(f"  External ({cond_name}): test={len(df)}")
        else:
            print(f"  External ({cond_name}): NOT FOUND (skipping)")
    return conditions


def bootstrap_pass(train, test_sets, n_bootstrap=100):
    """Run configuration-level bootstrap CIs for multiple test sets.

    Resamples configurations (all 20 rows of a configuration), not individual
    rows, so the interval reflects configuration-level Monte Carlo error. The
    test sets are held fixed.

    Parameters
    ----------
    train : training DataFrame (M1/M5)
    test_sets : dict of name -> (X_te, y_te, label)
    n_bootstrap : number of bootstrap iterations

    Returns
    -------
    results : dict of name -> {'r2': {'point': ..., 'ci_lower': ..., 'ci_upper': ...}, 'mae': ...}
    """
    groups = [g for _, g in train.groupby(GROUP_COLS, dropna=False)]
    X_tr, y_tr = prepare(train[FEATURES], train[TARGET].values)
    t0 = time.time()

    # Point estimate (full RF)
    rf_full = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=1, random_state=SEED
    )
    rf_full.fit(X_tr, y_tr)

    # Initialize storage
    names = list(test_sets.keys())
    n_tests = len(names)
    metrics = {name: {'r2': np.empty(n_bootstrap), 'mae': np.empty(n_bootstrap),
                      'rmse': np.empty(n_bootstrap)} for name in names}

    # Point estimates
    points = {}
    for name in names:
        X_te, y_te, _ = test_sets[name]
        yp = rf_full.predict(X_te)
        points[name] = {
            'r2': round(r2_score(y_te, yp), 4),
            'mae': round(mean_absolute_error(y_te, yp), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_te, yp)), 2),
        }

    # Bootstrap iterations: resample configurations, keeping each 20-row block
    rng = np.random.RandomState(SEED)
    for i in range(n_bootstrap):
        idx = rng.randint(0, len(groups), size=len(groups))
        sample = pd.concat([groups[j] for j in idx], ignore_index=True)
        Xb, yb = prepare(sample[FEATURES], sample[TARGET].values)

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=1, random_state=SEED + i
        )
        rf.fit(Xb, yb)

        for name in names:
            X_te, y_te, _ = test_sets[name]
            yp = rf.predict(X_te)
            metrics[name]['r2'][i] = r2_score(y_te, yp)
            metrics[name]['mae'][i] = mean_absolute_error(y_te, yp)
            metrics[name]['rmse'][i] = np.sqrt(mean_squared_error(y_te, yp))

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_bootstrap}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0

    # Build results with CIs
    results = {}
    for name in names:
        m = metrics[name]
        results[name] = {
            'r2': {
                'point': points[name]['r2'],
                'ci_lower': round(float(np.percentile(m['r2'], 2.5)), 4),
                'ci_upper': round(float(np.percentile(m['r2'], 97.5)), 4),
                'ci_90_lower': round(float(np.percentile(m['r2'], 5)), 4),
                'ci_90_upper': round(float(np.percentile(m['r2'], 95)), 4),
            },
            'mae': {
                'point': points[name]['mae'],
                'ci_lower': round(float(np.percentile(m['mae'], 2.5)), 2),
                'ci_upper': round(float(np.percentile(m['mae'], 97.5)), 2),
            },
            'rmse': {
                'point': points[name]['rmse'],
                'ci_lower': round(float(np.percentile(m['rmse'], 2.5)), 2),
                'ci_upper': round(float(np.percentile(m['rmse'], 97.5)), 2),
            },
        }

        print(f"\n  {name}:")
        print(f"    R2   = {points[name]['r2']:.4f}  "
              f"95% CI [{results[name]['r2']['ci_lower']:.4f}, "
              f"{results[name]['r2']['ci_upper']:.4f}]")
        print(f"    MAE  = {points[name]['mae']:.2f}  "
              f"95% CI [{results[name]['mae']['ci_lower']:.2f}, "
              f"{results[name]['mae']['ci_upper']:.2f}]")

    print(f"\n  Total: {elapsed:.0f}s for {n_bootstrap} iters x {n_tests} test sets")
    return results


def main():
    t0 = time.time()
    np.random.seed(SEED)

    print("=" * 60)
    print("BOOTSTRAP CI — Extended")
    print("=" * 60)

    # ── Load data ──
    print("\n--- Loading data ---")
    bin_train, bin_test = load_binary()
    cont_train, cont_test = load_continuous()
    X_bin_te, y_bin_te = prepare(bin_test[FEATURES], bin_test[TARGET].values)
    X_cont_te, y_cont_te = prepare(cont_test[FEATURES], cont_test[TARGET].values)
    ext_conds = load_external_conditions()

    # ── PASS 1: Binary regime ──
    #  Train on binary Y M1/M5, test on binary M6/M7 + external conditions
    test_sets_bin = {
        'Binary Y (held-out M6/M7)': (X_bin_te, y_bin_te, 'binary_heldout'),
    }
    for name, (X_te, y_te, n) in ext_conds.items():
        test_sets_bin[f'External: {name}'] = (X_te, y_te, n)

    print(f"\n{'=' * 60}")
    print("PASS 1: Binary Y regime (train M1/M5, test M6/M7 + external)")
    print(f"{'=' * 60}")
    results_bin = bootstrap_pass(bin_train, test_sets_bin, n_bootstrap=N_BOOTSTRAP)

    # ── PASS 2: Continuous regime ──
    test_sets_cont = {
        'Continuous Y (held-out M6/M7)': (X_cont_te, y_cont_te, 'cont_heldout'),
    }

    print(f"\n{'=' * 60}")
    print("PASS 2: Continuous Y regime (train M1/M5, test M6/M7)")
    print(f"{'=' * 60}")
    results_cont = bootstrap_pass(cont_train, test_sets_cont, n_bootstrap=N_BOOTSTRAP)

    # ── Merge and save ──
    results = {'binary_regime': results_bin, 'continuous_regime': results_cont}
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, 'meta_bootstrap_extended_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
