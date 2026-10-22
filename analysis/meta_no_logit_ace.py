"""
Meta-model without logit_ace: Diagnostic analysis of mechanical correlation.

Tests how meta-model performance changes when logit_ace (the strongest
predictor of its own bias) is removed from the feature set.
"""
import numpy as np
import pandas as pd
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

SEED = 20240625

FULL_FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

# Condition A: drop logit_ace only
NO_LOGIT_ACE = [f for f in FULL_FEATURES if f != 'logit_ace']
# Condition B: drop both logit_ace and lpm_ace (the two ACE estimates)
NO_ACE_ESTIMATES = [f for f in FULL_FEATURES if f not in ('logit_ace', 'lpm_x_coef')]

RF_PARAMS = dict(n_estimators=1000, max_depth=10, min_samples_leaf=5,
                 n_jobs=-1, random_state=SEED)


def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, '../results/raw/meta_training_data.parquet')
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows")
    return df


def prepare(X_df, y):
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def train_eval(train_df, test_df, features, label):
    """Train RF on given split, return metrics dict."""
    X_train, y_train = prepare(train_df[features], train_df['abs_bias_pct'].values)
    X_test, y_test = prepare(test_df[features], test_df['abs_bias_pct'].values)

    rf = RandomForestRegressor(**RF_PARAMS)
    t0 = time.time()
    rf.fit(X_train, y_train)
    t1 = time.time()

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importance
    imp = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    imp_idx = np.argsort(imp.importances_mean)[::-1]

    # Per-DAG breakdown
    per_dag = {}
    test_df_copy = test_df.copy()
    test_df_copy['pred'] = y_pred
    for model in sorted(test_df_copy['model_type'].unique()):
        sub = test_df_copy[test_df_copy['model_type'] == model]
        r2_m = r2_score(sub['abs_bias_pct'], sub['pred'])
        mae_m = mean_absolute_error(sub['abs_bias_pct'], sub['pred'])
        per_dag[model] = {'r2': r2_m, 'mae': mae_m, 'n': len(sub)}

    # Classification (high_bias > 5%)
    y_test_high = (y_test > 5).astype(int)
    y_pred_high = (y_pred > 5).astype(int)
    from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score
    acc = accuracy_score(y_test_high, y_pred_high)
    kappa = cohen_kappa_score(y_test_high, y_pred_high)
    bal_acc = balanced_accuracy_score(y_test_high, y_pred_high)

    # OLS baseline
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression().fit(X_train, y_train)
    ols_r2 = r2_score(y_test, ols.predict(X_test))
    ols_mae = mean_absolute_error(y_test, ols.predict(X_test))

    result = {
        'label': label,
        'fit_time': t1 - t0,
        'r2': r2, 'mae': mae, 'rmse': rmse,
        'n_features': X_train.shape[1],
        'ols_r2': ols_r2, 'ols_mae': ols_mae,
        'class_accuracy': acc, 'kappa': kappa, 'balanced_accuracy': bal_acc,
        'per_dag': per_dag,
        'feature_importance': [
            (features[i], imp.importances_mean[i], imp.importances_std[i])
            for i in imp_idx
        ],
    }

    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X_train.shape[1]}")
    print(f"  RF R2 = {r2:.4f} | MAE = {mae:.2f}pp | RMSE = {rmse:.2f}pp")
    print(f"  OLS R2 = {ols_r2:.4f} | OLS MAE = {ols_mae:.2f}pp")
    print(f"  Fit time: {t1-t0:.1f}s")
    print(f"  Class acc = {acc:.4f} | kappa = {kappa:.4f} | balanced acc = {bal_acc:.4f}")

    print(f"\n  Feature importances:")
    for rank, (name, mean_imp, std_imp) in enumerate(result['feature_importance'][:10]):
        print(f"    {rank+1}. {name:<20} {mean_imp:.4f} ± {std_imp:.4f}")

    print(f"\n  Per-DAG R2:")
    for m in sorted(per_dag.keys()):
        print(f"    {m}: R2={per_dag[m]['r2']:.4f}, MAE={per_dag[m]['mae']:.2f}pp (n={per_dag[m]['n']})")

    return result


def split_by_model(df, train_models, test_models):
    train = df[df['model_type'].isin(train_models)].copy()
    test = df[df['model_type'].isin(test_models)].copy()
    return train, test


def run():
    t0 = time.time()
    print("=" * 65)
    print("DIAGNOSTIC: META-MODEL WITHOUT logit_ace")
    print("=" * 65)

    df = load_data()
    print(f"High-bias rate: {df['high_bias'].mean()*100:.1f}%")
    print(f"Mean |bias|: {df['abs_bias_pct'].mean():.1f}%")

    # ── Split definitions ──
    splits = {
        'Forward (train M1/M5 → test M6/M7)': (['M1', 'M5'], ['M6', 'M7']),
        'Reverse (train M6/M7 → test M1/M5)': (['M6', 'M7'], ['M1', 'M5']),
    }

    # ── Feature sets ──
    feature_sets = [
        ('Full (15 features)', FULL_FEATURES),
        ('No logit_ace (14 features)', NO_LOGIT_ACE),
        ('No ACE estimates (13 features)', NO_ACE_ESTIMATES),
    ]

    all_results = []

    for split_name, (train_models, test_models) in splits.items():
        print(f"\n{'=' * 65}")
        print(f"SPLIT: {split_name}")
        print(f"{'=' * 65}")
        train_df, test_df = split_by_model(df, train_models, test_models)
        print(f"  Train: {len(train_df)} rows ({', '.join(train_models)})")
        print(f"  Test:  {len(test_df)} rows ({', '.join(test_models)})")

        for label, features in feature_sets:
            result = train_eval(train_df, test_df, features, label)
            result['split'] = split_name
            all_results.append(result)

    # ── Summary tables ──
    print(f"\n{'=' * 65}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 65}")

    for split_name in splits:
        print(f"\n{split_name}:")
        print(f"{'Feature set':<28} {'R2':>8} {'MAE(pp)':>10} {'OLS_R2':>8} {'Kappa':>8}")
        print(f"{'─'*28} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")
        for r in all_results:
            if r['split'] == split_name:
                print(f"{r['label']:<28} {r['r2']:>8.4f} {r['mae']:>8.2f}pp {r['ols_r2']:>8.4f} {r['kappa']:>8.4f}")

    # ── Per-DAG comparison (forward split, all feature sets) ──
    print(f"\n{'=' * 65}")
    print("PER-DAG R2 COMPARISON (Forward split)")
    print(f"{'=' * 65}")
    dag_models = sorted(df['model_type'].unique())
    print(f"{'Feature set':<28} {'':<5} ", end="")
    for m in dag_models:
        print(f"{m:>8}", end="")
    print()
    print(f"{'─'*28} {'':<5} ", end="")
    for _ in dag_models:
        print(f"{'─'*8}", end="")
    print()
    for r in all_results:
        if r['split'] == 'Forward (train M1/M5 → test M6/M7)':
            print(f"{r['label']:<28} {'':>5} ", end="")
            for m in dag_models:
                val = r['per_dag'].get(m, {})
                print(f"{val.get('r2', 0):>8.4f}", end="")
            print()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return all_results


if __name__ == '__main__':
    results = run()
