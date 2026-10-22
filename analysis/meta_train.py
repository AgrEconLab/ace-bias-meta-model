"""
Meta-model: Training + Diagnostic Rule Derivation

Trains two complementary meta-models on the simulated training data:
  1. Random Forest -- prediction accuracy benchmark, feature importance ranking
  2. Decision Tree -- interpretable diagnostic rule for practitioner use

Then derives a simple scoring rule for estimating bias risk from observable data.

Predictors (all observable without knowing true ACE):
  - n, y_mean, x_var, logit_ace, lpm_ace, logit_lpm_diff
  - unadj_ace, adj_unadj_diff, z_coef_mag, z_x_coef_mag
  - y_pred_min, y_pred_max, y_pred_range, lpm_x_coef, y_model_r2

Target: abs_bias_pct (|bias| of logit g-computation in percent)
"""

import numpy as np
import pandas as pd
import sys, os, time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, cohen_kappa_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.inspection import permutation_importance

SEED = 20240625

# Features used for meta-prediction
FEATURES_OBSERVABLE = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

# Features for reduced interpretable model
FEATURES_REDUCED = [
    'logit_lpm_diff', 'adj_unadj_diff', 'y_mean',
    'n', 'z_coef_mag',
]


def load_data():
    """Load training data from parquet or CSV."""
    base = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base, '../results/raw/meta_training_data.parquet')
    csv_path = parquet_path.replace('.parquet', '.csv')

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows from parquet")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
    else:
        raise FileNotFoundError(f"No training data found at {parquet_path}")

    return df


def prepare_data(df, features=None, target='abs_bias_pct'):
    """Prepare feature matrix and target vector."""
    if features is None:
        features = FEATURES_OBSERVABLE
    X = df[features].copy()
    y = df[target].values

    # Handle any infinities (logit_lpm_diff can be large)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, y, features


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train RF regressor for |bias| prediction."""
    print("\n" + "─" * 60)
    print("MODEL A: RANDOM FOREST REGRESSOR")
    print("─" * 60)

    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED, verbose=0
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    t1 = time.time()

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  Fit in {t1 - t0:.1f}s")
    print(f"  R2 (train): {r2_train:.4f}")
    print(f"  R2 (test):  {r2_test:.4f}")
    print(f"  MAE (test): {mae_test:.2f}pp")
    print(f"  RMSE (test): {rmse_test:.2f}pp")

    # Feature importance
    imp = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    imp_idx = np.argsort(imp.importances_mean)[::-1]

    print(f"\n  Top-10 Feature Importances (permutation):")
    print(f"  {'Rank':<6} {'Feature':<22} {'Importance':>10} {'SD':>8}")
    print(f"  {'─'*6} {'─'*22} {'─'*10} {'─'*8}")
    for rank, idx in enumerate(imp_idx[:10]):
        print(f"  {rank + 1:<6} {feature_names[idx]:<22} "
              f"{imp.importances_mean[idx]:>8.4f}  {imp.importances_std[idx]:>6.4f}")

    # Classification accuracy for high_bias (|bias| > 5%)
    y_test_high = (y_test > 5).astype(int)
    y_pred_high = (y_pred_test > 5).astype(int)
    acc = accuracy_score(y_test_high, y_pred_high)
    prec = precision_score(y_test_high, y_pred_high, zero_division=0)
    rec = recall_score(y_test_high, y_pred_high, zero_division=0)
    cm = confusion_matrix(y_test_high, y_pred_high)

    print(f"\n  High-bias classification (>5% |bias|):")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    Cohen's kappa: {cohen_kappa_score(y_test_high, y_pred_high):.3f}")
    print(f"    Balanced accuracy: {balanced_accuracy_score(y_test_high, y_pred_high):.3f}")
    print(f"    Confusion matrix:")
    print(f"      TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]}")

    return rf, {
        'r2_train': r2_train, 'r2_test': r2_test,
        'mae_test': mae_test, 'rmse_test': rmse_test,
        'feature_importance': list(zip(feature_names, imp.importances_mean)),
        'high_bias_accuracy': acc,
        'high_bias_precision': prec,
        'high_bias_recall': rec,
    }


def train_decision_tree(X_train, y_train, X_test, y_test, feature_names, max_depth=3):
    """Train an interpretable decision tree for diagnostic rule."""
    print("\n" + "─" * 60)
    print(f"MODEL B: DECISION TREE (max_depth={max_depth})")
    print("─" * 60)

    dt = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=10,
        random_state=SEED
    )
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"  R2 (train): {r2_train:.4f}")
    print(f"  R2 (test):  {r2_test:.4f}")
    print(f"  MAE (test): {mae_test:.2f}pp")

    # Print tree as text
    print(f"\n  Diagnostic Rule (Decision Tree):")
    print(f"  {'─' * 50}")
    rule_text = export_text(dt, feature_names=list(feature_names),
                            show_weights=True, decimals=3)
    for line in rule_text.strip().split('\n'):
        print(f"    {line}")

    # Also fit a classifier on high_bias
    print(f"\n  Classification version (predict |bias| > 5%):")
    dtc = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10,
                                 random_state=SEED)
    y_train_high = (y_train > 5).astype(int)
    y_test_high = (y_test > 5).astype(int)
    dtc.fit(X_train, y_train_high)
    y_pred_high = dtc.predict(X_test)
    acc = accuracy_score(y_test_high, y_pred_high)
    prec = precision_score(y_test_high, y_pred_high, zero_division=0)
    rec = recall_score(y_test_high, y_pred_high, zero_division=0)
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")

    dtc_rule = export_text(dtc, feature_names=list(feature_names),
                           show_weights=True, decimals=3)
    for line in dtc_rule.strip().split('\n'):
        print(f"    {line}")

    return dt, {
        'r2_train': r2_train, 'r2_test': r2_test, 'mae_test': mae_test,
        'tree_text': rule_text,
        'classifier_accuracy': acc,
    }


def train_logistic_model(X_train, y_train, X_test, y_test, feature_names):
    """Train a simple logistic regression for high_bias prediction."""
    print("\n" + "─" * 60)
    print("MODEL C: LOGISTIC REGRESSION (high-bias classifier)")
    print("─" * 60)

    y_train_high = (y_train > 5).astype(int)
    y_test_high = (y_test > 5).astype(int)

    lr = LogisticRegression(max_iter=5000, random_state=SEED, C=1.0)
    lr.fit(X_train, y_train_high)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test_high, y_pred)
    prec = precision_score(y_test_high, y_pred, zero_division=0)
    rec = recall_score(y_test_high, y_pred, zero_division=0)

    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")

    # Coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': lr.coef_[0],
        'abs_coef': np.abs(lr.coef_[0]),
        'sign': np.sign(lr.coef_[0]),
    }).sort_values('abs_coef', ascending=False)

    print(f"\n  Coefficients (sorted by |coef|):")
    print(f"  {'Feature':<22} {'Coef':>8} {'Sign':>6}")
    print(f"  {'─'*22} {'─'*8} {'─'*6}")
    for _, row in coef_df.iterrows():
        sign = '+' if row['coef'] > 0 else '-'
        print(f"  {row['feature']:<22} {row['coef']:>+8.4f} {sign:>6}")

    # Score rule: probability = 1 / (1 + exp(-(intercept + sum(coef_i * x_i)))
    intercept = lr.intercept_[0]
    print(f"\n  Score function:")
    print(f"    logit(P(high_bias)) = {intercept:+.4f}")
    for _, row in coef_df.iterrows():
        print(f"      {row['coef']:+.4f} x {row['feature']}")

    return lr, {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'coefficients': coef_df,
        'intercept': intercept,
    }


def compute_baseline_metrics(df, y_test):
    """Compute baseline metrics for comparison."""
    print("\n" + "─" * 60)
    print("BASELINE COMPARISONS")
    print("─" * 60)

    # Baseline 1: always predict mean bias
    mean_bias = df['abs_bias_pct'].mean()
    baseline_mae = np.mean(np.abs(y_test - mean_bias))
    print(f"  Baseline 1 (predict mean): MAE = {baseline_mae:.2f}pp")

    # Baseline 2: always trust logit (predict bias = 0)
    baseline2_mae = np.mean(np.abs(y_test - 0))
    print(f"  Baseline 2 (always 0 bias): MAE = {baseline2_mae:.2f}pp")

    # Baseline 3: choose estimator by rule
    # "If LPM and logit agree, trust either; if they disagree, logit is better"
    # This is just a heuristic -- compute the error of always picking logit
    y_test_high = (y_test > 5).astype(int)
    always_logit_acc = 1 - y_test_high.mean()
    print(f"  Baseline 3 (always trust logit): accuracy = {always_logit_acc:.3f}")
    print(f"    (since {y_test_high.mean()*100:.1f}% of test cases have high bias)")

    return {
        'baseline_mae_mean': baseline_mae,
        'baseline_mae_zero': baseline2_mae,
        'always_logit_accuracy': always_logit_acc,
    }


def split_by_model(df, train_models=None, test_models=None):
    """Split data by model family for out-of-model generalization test."""
    if train_models is None:
        train_models = ['M1', 'M5']
    if test_models is None:
        test_models = ['M6', 'M7']

    train = df[df['model_type'].isin(train_models)].copy()
    test = df[df['model_type'].isin(test_models)].copy()
    return train, test


def split_random(df, test_frac=0.2):
    """Random train-test split."""
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_frac, random_state=SEED)
    return train, test


def run_pipeline():
    t0 = time.time()
    print("=" * 70)
    print("META-MODEL: TRAINING + DIAGNOSTIC RULE DERIVATION")
    print("=" * 70)

    # ── Load data ──
    df = load_data()
    print(f"\nRows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"High-bias rate: {df['high_bias'].mean()*100:.1f}%")
    print(f"Mean |bias|: {df['abs_bias_pct'].mean():.2f}pp")
    print(f"LPM wins rate: {df['lpm_wins'].mean()*100:.1f}%")

    # ── Split: hold out M6 + M7 for out-of-model testing ──
    print(f"\n{'=' * 70}")
    print("SPLIT: Train on M1/M5, Test on M6/M7 (out-of-model generalization)")
    print(f"{'=' * 70}")
    train_df, test_df = split_by_model(df, train_models=['M1', 'M5'],
                                        test_models=['M6', 'M7'])
    print(f"  Train: {len(train_df)} rows (M1/M5)")
    print(f"  Test:  {len(test_df)} rows (M6/M7)")

    X_train, y_train, feat = prepare_data(train_df, FEATURES_OBSERVABLE)
    X_test, y_test, _ = prepare_data(test_df, FEATURES_OBSERVABLE)

    # Sanity check
    print(f"\n  Feature matrix: {X_train.shape[1]} features x {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"  Target: |bias%| -- range [{y_train.min():.1f}, {y_train.max():.1f}], "
          f"mean {y_train.mean():.1f}")

    # ── Baselines ──
    baselines = compute_baseline_metrics(train_df, y_test)

    # ── Model A: Random Forest ──
    rf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, feat)

    # ── Model B: Decision Tree (interpretable) ──
    dt, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test, feat, max_depth=3)

    # ── Model C: Logistic Regression ──
    lr, lr_metrics = train_logistic_model(X_train, y_train, X_test, y_test, feat)

    # ── Random-split evaluation (within-model generalization) ──
    print("\n" + "─" * 60)
    print("RANDOM-SPLIT EVALUATION (80/20, within-model)")
    print("─" * 60)

    train_r, test_r = split_random(df, test_frac=0.2)
    Xr_train, yr_train, _ = prepare_data(train_r, FEATURES_OBSERVABLE)
    Xr_test, yr_test, _ = prepare_data(test_r, FEATURES_OBSERVABLE)

    rf_r = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED
    )
    rf_r.fit(Xr_train, yr_train)
    r2_random = r2_score(yr_test, rf_r.predict(Xr_test))
    mae_random = mean_absolute_error(yr_test, rf_r.predict(Xr_test))

    # Per-model breakdown
    print(f"  Overall R2: {r2_random:.4f}, MAE: {mae_random:.2f}pp")
    test_r['rf_pred'] = rf_r.predict(Xr_test)
    for model in sorted(test_r['model_type'].unique()):
        subset = test_r[test_r['model_type'] == model]
        r2_m = r2_score(subset['abs_bias_pct'].values, subset['rf_pred'].values)
        mae_m = mean_absolute_error(subset['abs_bias_pct'].values, subset['rf_pred'].values)
        print(f"  {model}: R2={r2_m:.4f}, MAE={mae_m:.2f}pp (n={len(subset)})")

    # ── Reduced model (only top features) ──
    print("\n" + "─" * 60)
    print("REDUCED MODEL (top observable features)")
    print("─" * 60)

    Xr_train, yr_train, rfeat = prepare_data(train_df, FEATURES_REDUCED)
    Xr_test, yr_test, _ = prepare_data(test_df, FEATURES_REDUCED)

    dt_r = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=SEED)
    dt_r.fit(Xr_train, yr_train)
    r2_r = r2_score(yr_test, dt_r.predict(Xr_test))
    mae_r = mean_absolute_error(yr_test, dt_r.predict(Xr_test))
    print(f"  Reduced Decision Tree R2 (test): {r2_r:.4f}")
    print(f"  Reduced Decision Tree MAE (test): {mae_r:.2f}pp")
    print(f"\n  Reduced Tree Rule:")
    rule_r = export_text(dt_r, feature_names=rfeat, show_weights=True, decimals=3)
    for line in rule_r.strip().split('\n'):
        print(f"    {line}")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n{'Model':<30} {'R2(test)':>10} {'MAE':>8} {'Acc(>5%)':>10}")
    print(f"{'─'*30} {'─'*10} {'─'*8} {'─'*10}")
    print(f"{'Baseline (predict mean)':<30} {'--':>10} {baselines['baseline_mae_mean']:>7.2f}pp {'--':>10}")
    print(f"{'Baseline (always 0)':<30} {'--':>10} {baselines['baseline_mae_zero']:>7.2f}pp {'--':>10}")
    print(f"{'Baseline (always trust logit)':<30} {'--':>10} {'--':>8} "
          f"{baselines['always_logit_accuracy']:>9.3f}")
    print(f"{'Random Forest':<30} {rf_metrics['r2_test']:>9.4f} "
          f"{rf_metrics['mae_test']:>7.2f}pp {rf_metrics['high_bias_accuracy']:>9.3f}")
    print(f"{f'Decision Tree (depth=3)':<30} {dt_metrics['r2_test']:>9.4f} "
          f"{dt_metrics['mae_test']:>7.2f}pp {dt_metrics['classifier_accuracy']:>9.3f}")
    print(f"{'Logistic Regression':<30} {'--':>10} {'--':>8} "
          f"{lr_metrics['accuracy']:>9.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("=" * 70)

    return df, rf, dt, lr, {
        'rf_metrics': rf_metrics,
        'dt_metrics': dt_metrics,
        'lr_metrics': lr_metrics,
        'baselines': baselines,
    }


if __name__ == '__main__':
    run_pipeline()
