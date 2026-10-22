"""
Bootstrap CI for external validation conditions.

Loads saved parquet data from meta_external_val.py, trains RF on binary Y
M1/M5, runs 100 bootstrap iterations, and computes percentile CIs for
R², MAE, and RMSE on each external condition.

Requires: results/raw/meta_ext_*.parquet (from meta_external_val.py)
Outputs:   results/raw/meta_bootstrap_external_results.json
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

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, '../results/raw')


def prepare(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median()
    return X.fillna(med), y


def load_training_data():
    """Load binary Y M1/M5 training data (same as meta_bootstrap_extended.py)."""
    path = os.path.join(RAW_DIR, 'meta_training_data.parquet')
    df = pd.read_parquet(path)
    train = df[df['model_type'].isin(['M1', 'M5'])].copy()
    X_tr, y_tr = prepare(train[FEATURES], train[TARGET].values)
    print(f"  Training data: {len(train)} rows (M1/M5)")
    return X_tr, y_tr


def load_external_conditions():
    """Load saved external validation parquet files."""
    conditions = {}
    for cond_name in ['1_Probit', '2_ExtendedRange', '3_UniformZ']:
        path = os.path.join(RAW_DIR, f'meta_ext_{cond_name}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            X_te, y_te = prepare(df[FEATURES], df[TARGET].values)
            conditions[cond_name] = (X_te, y_te, len(df))
            print(f"  {cond_name}: {len(df)} rows")
        else:
            print(f"  {cond_name}: NOT FOUND (skipping)")
    return conditions


def bootstrap_pass(X_tr, y_tr, test_sets, n_bootstrap=100):
    """Run bootstrap CIs for multiple test sets from a single training set.

    Parameters
    ----------
    X_tr, y_tr : training data (M1/M5, binary Y)
    test_sets  : dict of name -> (X_te, y_te, n_rows)
    n_bootstrap: number of bootstrap iterations

    Returns
    -------
    results    : dict of name -> metrics dict with point and CI
    """
    n_train = X_tr.shape[0]
    t0 = time.time()

    # Point estimate (full RF)
    rf_full = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=2, random_state=SEED
    )
    rf_full.fit(X_tr, y_tr)

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

    # Bootstrap iterations
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n_train, size=n_train)
        Xb = X_tr.iloc[idx] if hasattr(X_tr, 'iloc') else X_tr[idx]
        yb = y_tr[idx]

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=2, random_state=SEED + i
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

    # Build results
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
    print("BOOTSTRAP CI — External Validation")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    X_tr, y_tr = load_training_data()
    ext_conds = load_external_conditions()

    if not ext_conds:
        print("No external validation data found. Run meta_external_val.py first.")
        return

    # Bootstrap
    print(f"\n{'=' * 60}")
    print("Bootstrap (binary Y M1/M5 -> external conditions)")
    print(f"{'=' * 60}")
    results = bootstrap_pass(X_tr, y_tr, ext_conds, n_bootstrap=N_BOOTSTRAP)

    # Save
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, 'meta_bootstrap_external_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
