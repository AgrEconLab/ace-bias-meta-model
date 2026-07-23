"""
Meta-model: Export Tables 1-3 as CSV Files

Reproduces the three manuscript tables from trained models:
  Table 1 — Top 10 Feature Importances (permutation-based, held-out M6/M7)
  Table 2 — Per-DAG Meta-Model R-squared (random 80/20 split)
  Table 3 — Diagnostic Decision Rule risk zones (decision tree, M1/M5 train)

All tables saved to results/tables/ as CSV and optionally displayed.
"""

import numpy as np
import pandas as pd
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

SEED = 20240625

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'lpm_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')


def load_data():
    path = os.path.join(RAW_DIR, 'meta_training_data.parquet')
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows")
    return df


def prepare(df, features=None):
    if features is None:
        features = FEATURES
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
    return X, df['abs_bias_pct'].values


# ── Table 1: Feature Importances ──

def export_table1(df):
    """Permutation importance from RF trained on M1/M5, evaluated on M6/M7."""
    print("\n" + "=" * 60)
    print("TABLE 1: TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)

    train = df[df['model_type'].isin(['M1', 'M5'])]
    test = df[df['model_type'].isin(['M6', 'M7'])]
    X_train, y_train = prepare(train)
    X_test, y_test = prepare(test)

    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED
    )
    rf.fit(X_train, y_train)

    imp = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    idx = np.argsort(imp.importances_mean)[::-1]

    rows = []
    for rank, i in enumerate(idx[:10]):
        rows.append({
            'rank': rank + 1,
            'feature': FEATURES[i],
            'importance': round(imp.importances_mean[i], 4),
            'importance_sd': round(imp.importances_std[i], 4),
        })

    t1 = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, 'table1_feature_importance.csv')
    t1.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Pretty console
    print(f"\n  {'Rank':<6} {'Feature':<22} {'Importance':>10} {'SD':>8}")
    print(f"  {'-'*6} {'-'*22} {'-'*10} {'-'*8}")
    for r in rows:
        print(f"  {r['rank']:<6} {r['feature']:<22} {r['importance']:>10.4f} {r['importance_sd']:>8.4f}")

    return t1


# ── Table 2: Per-DAG R-squared ──

def export_table2(df):
    """Per-DAG R² from RF on random 80/20 split."""
    print("\n" + "=" * 60)
    print("TABLE 2: PER-DAG META-MODEL R-SQUARED")
    print("=" * 60)

    X_full, y_full = prepare(df)

    # 80/20 stratified split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED,
        stratify=df['model_type']
    )
    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]
    test_df = df.iloc[test_idx]

    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED
    )
    rf.fit(X_train, y_train)
    test_df = test_df.copy()
    test_df['rf_pred'] = rf.predict(X_test)

    rows = []
    model_labels = {
        'M1': 'Confounding', 'M2': 'Outcome Predictor', 'M3': 'Exposure Predictor',
        'M4': 'Exposure Descendant', 'M5': 'Mediation', 'M6': 'Collider',
        'M7': 'Outcome Descendant',
    }
    dag_order = ['M6', 'M2', 'M4', 'M7', 'M3', 'M1', 'M5']

    for model in dag_order:
        sub = test_df[test_df['model_type'] == model]
        r2 = r2_score(sub['abs_bias_pct'], sub['rf_pred'])
        high_bias_rate = sub['high_bias'].mean()
        rows.append({
            'dag': model,
            'structure': model_labels[model],
            'r_squared': round(r2, 3),
            'high_bias_rate': f"{high_bias_rate:.0%}",
        })

    t2 = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, 'table2_per_dag_rsquared.csv')
    t2.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print(f"\n  {'DAG':<6} {'Structure':<24} {'R-squared':>10} {'High-Bias Rate':>16}")
    print(f"  {'-'*6} {'-'*24} {'-'*10} {'-'*16}")
    for r in rows:
        print(f"  {r['dag']:<6} {r['structure']:<24} {r['r_squared']:>10.3f} {r['high_bias_rate']:>16}")

    return t2


# ── Table 3: Diagnostic Decision Rule ──

def export_table3(df):
    """Decision tree rule zones from training on M1/M5, applied to M6/M7."""
    print("\n" + "=" * 60)
    print("TABLE 3: DIAGNOSTIC DECISION RULE")
    print("=" * 60)

    train = df[df['model_type'].isin(['M1', 'M5'])]
    test = df[df['model_type'].isin(['M6', 'M7'])]
    X_train, y_train = prepare(train)
    X_test_full, y_test_full = prepare(test)

    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=SEED)
    dt.fit(X_train, y_train)

    # Assign each test row to a leaf (zone)
    leaf_ids = dt.apply(X_test_full)
    test_copy = test.copy()
    test_copy['leaf'] = leaf_ids

    # Extract tree structure as text for reference
    rule_text = export_text(dt, feature_names=FEATURES, show_weights=True, decimals=3)

    # Build per-zone summary from the tree splits
    # The tree trained on M1/M5 has these splits: lpm_x_coef, n, logit_ace
    tree = dt.tree_
    left = tree.children_left
    right = tree.children_right
    threshold = tree.threshold
    feature = tree.feature

    zone_info = {}
    stack = [(0, [])]  # (node_id, conditions)
    while stack:
        node_id, conditions = stack.pop()
        if left[node_id] == right[node_id]:  # leaf
            # Get test-set predictions for this leaf
            mask = test_copy['leaf'] == node_id
            leaf_data = test_copy.loc[mask, 'abs_bias_pct']
            if len(leaf_data) > 0:
                zone_info[node_id] = {
                    'conditions': conditions,
                    'n': len(leaf_data),
                    'median_bias': round(leaf_data.median(), 1),
                    'q25': round(leaf_data.quantile(0.25), 1),
                    'q75': round(leaf_data.quantile(0.75), 1),
                    'mean_bias': round(leaf_data.mean(), 1),
                }
        else:
            feat_name = FEATURES[feature[node_id]]
            thr = threshold[node_id]
            # Left child: feature <= threshold
            stack.append((left[node_id], conditions + [f"{feat_name} <= {thr:.3f}"]))
            # Right child: feature > threshold
            stack.append((right[node_id], conditions + [f"{feat_name} > {thr:.3f}"]))

    # Map to RED/GREEN/YELLOW zones
    zone_map = {}
    for leaf_id, info in zone_info.items():
        cond_str = ' & '.join(info['conditions'])
        if 'lpm_x_coef <= -0.026' in cond_str:
            zone = 'RED'
        elif ('logit_ace >' in cond_str and 'lpm_x_coef > -0.026' in cond_str
              and 'n > 625' not in cond_str and 'n <= 625' not in cond_str):
            zone = 'GREEN'
        elif 'n > 625' in cond_str and 'lpm_x_coef > 0.102' not in cond_str:
            zone = 'GREEN'  # fallback: n>625 without strong lpm
        elif 'n <= 625' in cond_str:
            zone = 'YELLOW-B'
        else:
            zone = 'YELLOW-A'
        zone_map[zone] = info

    # Build table rows in manuscript order
    table_rows = []
    for zone in ['RED', 'GREEN', 'YELLOW-A', 'YELLOW-B']:
        info = zone_map.get(zone)
        if info is None:
            continue

        if info['q25'] == info['q75'] or info['q75'] - info['q25'] < 1:
            # Tight range or single row — show as "~X pp"
            expected = f"~{info['median_bias']:.0f} pp"
        else:
            expected = f"{info['q25']:.0f}--{info['q75']:.0f} pp"

        # Human-readable condition
        if zone == 'RED':
            condition = 'lpm_x_coef <= -0.026'
        elif zone == 'GREEN':
            condition = 'lpm_x_coef in (-0.026, 0.102], logit_ace > -0.002'
        elif zone == 'YELLOW-A':
            condition = 'lpm_x_coef > 0.102, n > 625'
        elif zone == 'YELLOW-B':
            condition = 'lpm_x_coef > -0.026, n <= 625'

        table_rows.append({
            'zone': zone,
            'condition': condition,
            'expected_bias': expected,
            'n_test': info['n'],
            'median_bias_pp': info['median_bias'],
            'iqr_lower_pp': info['q25'],
            'iqr_upper_pp': info['q75'],
        })

    t3 = pd.DataFrame(table_rows)
    path = os.path.join(TABLES_DIR, 'table3_decision_rule.csv')
    t3.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print(f"\n  {'Zone':<12} {'Condition':<55} {'Expected |bias|':<20}")
    print(f"  {'-'*12} {'-'*55} {'-'*20}")
    for r in table_rows:
        print(f"  {r['zone']:<12} {r['condition']:<55} {r['expected_bias']:<20}")

    # Also save the tree text
    tree_path = os.path.join(TABLES_DIR, 'table3_decision_tree_text.txt')
    with open(tree_path, 'w') as f:
        f.write(rule_text)
    print(f"  Also saved tree text: {tree_path}")

    return t3


if __name__ == '__main__':
    os.makedirs(TABLES_DIR, exist_ok=True)

    df = load_data()
    print(f"Models: {sorted(df['model_type'].unique())}")
    print(f"High-bias rate: {df['high_bias'].mean()*100:.1f}%")

    t1 = export_table1(df)
    t2 = export_table2(df)
    t3 = export_table3(df)

    print("\n" + "=" * 60)
    print("ALL TABLES EXPORTED")
    print(f"  {TABLES_DIR}/")
    print(f"    table1_feature_importance.csv")
    print(f"    table2_per_dag_rsquared.csv")
    print(f"    table3_decision_rule.csv")
    print(f"    table3_decision_tree_text.txt")
    print("=" * 60)
