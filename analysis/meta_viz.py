"""
Meta-model: Diagnostic Visualizations

Generates 4 figures for the meta-model diagnostic rule analysis:
  1. Feature importance bar chart
  2. Predicted vs actual bias scatter (RF)
  3. Bias distribution by model family
  4. Partial dependence plots for top-3 features
"""

import numpy as np
import pandas as pd
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_absolute_error

from config import FIGURES_DIR

SEED = 20240625

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'lpm_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

plt.rcParams.update({
    'figure.dpi': 300,
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# JPG export settings
JPG_DPI = 400


def _save_formats(fig, full_stem, tight=True):
    """Save figure as both PDF and high-res JPG. full_stem is path without extension."""
    pdf_path = str(full_stem) + '.pdf'
    jpg_path = str(full_stem) + '.jpg'
    kwargs = {'bbox_inches': 'tight'} if tight else {}
    fig.savefig(pdf_path, **kwargs)
    fig.savefig(jpg_path, dpi=JPG_DPI, **kwargs)
    print(f"  [OK] {os.path.basename(pdf_path)} + {os.path.basename(jpg_path)}")

MODEL_COLORS = {'M1': '#2ecc71', 'M5': '#f39c12', 'M6': '#e74c3c', 'M7': '#3498db'}
MODEL_LABELS = {'M1': 'M1: Confounding', 'M5': 'M5: Mediation',
                'M6': 'M6: Collider', 'M7': 'M7: Outcome Desc.'}


def load_and_prepare():
    """Load data and return full + held-out splits."""
    base = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base, '../results/raw/meta_training_data.parquet')
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['abs_bias_pct'].values
    y_high = df['high_bias'].values

    # Held-out split for Figure 1
    train_mask = df['model_type'].isin(['M1', 'M5'])
    test_mask = df['model_type'].isin(['M6', 'M7'])
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    df_te = df[test_mask].copy()

    return df, X, y, y_high, df_te, X_tr, y_tr, X_te, y_te


def fig_feature_importance(df, X, y, filename='fig_meta_importance'):
    """Bar chart of permutation feature importance from RF."""
    print(f"\n[1/4] Feature importance ({filename})")

    rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                               min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf.fit(X, y)

    from sklearn.inspection import permutation_importance
    imp = permutation_importance(rf, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_idx = np.argsort(imp.importances_mean)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    top_n = min(12, len(FEATURES))
    names = [FEATURES[i] for i in imp_idx[:top_n]]
    scores = imp.importances_mean[imp_idx[:top_n]]
    sds = imp.importances_std[imp_idx[:top_n]]

    # Feature labels with descriptions
    label_map = {
        'logit_ace': 'Logit ACE estimate',
        'lpm_ace': 'LPM ACE estimate',
        'lpm_x_coef': 'LPM X coefficient',
        'n': 'Sample size (N)',
        'unadj_ace': 'Unadj. ACE estimate',
        'logit_lpm_diff': '|Logit ACE - LPM ACE|',
        'y_pred_min': 'Min predicted P(Y=1)',
        'y_pred_range': 'Predicted probability range',
        'y_pred_max': 'Max predicted P(Y=1)',
        'x_var': 'Treatment variance (Var X)',
        'z_coef_mag': '|Z coef| in Y~X+Z (logit)',
        'y_model_r2': 'Y model R-squared',
    }
    display_names = [label_map.get(n, n) for n in names]

    ax.barh(range(top_n), scores[::-1], xerr=sds[::-1], capsize=3,
            color='#2c3e50', alpha=0.85, height=0.6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(display_names[::-1], fontsize=10)
    ax.set_xlabel('Permutation Importance (|bias| prediction)')
    ax.set_title('Top Predictors of Logit ACE |Bias|', fontweight='bold')
    ax.axvline(0, color='gray', lw=0.5)

    # Add R2 annotation
    r2 = r2_score(y, rf.predict(X))
    ax.text(0.95, 0.05, f'Within-sample R² = {r2:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, fontstyle='italic')

    plt.tight_layout()
    _save_formats(fig, FIGURES_DIR / filename, tight=True)
    print(f"  [OK] Saved {filename}")
    plt.close(fig)

    return rf


def fig_predicted_vs_actual(df_te, X_tr, y_tr, X_te, y_te, filename='fig_meta_pred_vs_actual'):
    """Scatter plot of predicted vs actual |bias|, colored by model type.

    Uses held-out split (train M1/M5, test M6/M7) so the R2 annotation
    matches the manuscript's primary evaluation metric (R2=0.633).
    """
    print(f"[2/4] Predicted vs actual ({filename})")

    rf_ho = RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_leaf=5,
        n_jobs=2, random_state=SEED
    )
    rf_ho.fit(X_tr, y_tr)
    y_pred = rf_ho.predict(X_te)

    fig, ax = plt.subplots(figsize=(9, 7))

    for model in ['M6', 'M7']:
        mask = df_te['model_type'] == model
        ax.scatter(y_pred[mask], y_te[mask], c=MODEL_COLORS[model],
                   label=MODEL_LABELS[model], alpha=0.3, s=8, edgecolors='none')

    ax.plot([0, y_te.max()], [0, y_te.max()], 'k--', lw=1, alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Predicted |bias| (%)')
    ax.set_ylabel('Actual |bias| (%)')
    ax.set_title('Random Forest: Predicted vs Actual |bias| (held-out M6/M7)', fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)

    # R2 annotation (held-out)
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    ax.text(0.95, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f}pp',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontstyle='italic',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    _save_formats(fig, FIGURES_DIR / filename, tight=True)
    print(f"  [OK] Saved {filename}")
    plt.close(fig)


def fig_bias_by_model(df, filename='fig_meta_bias_by_model'):
    """Bias distribution by model family, highlighting structural patterns."""
    print(f"[3/4] Bias by model ({filename})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: |bias| distribution by model
    ax = axes[0]
    data_points = []
    labels = []
    colors = []
    for model in ['M1', 'M5', 'M6', 'M7']:
        vals = df[df['model_type'] == model]['abs_bias_pct'].values
        data_points.append(vals)
        labels.append(MODEL_LABELS[model])
        colors.append(MODEL_COLORS[model])

    bp = ax.boxplot(data_points, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', lw=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('|bias| (%)')
    ax.set_title('A: |Bias| Distribution by DAG Structure', fontweight='bold')
    ax.set_yscale('symlog')  # log-like but handles zeros

    # Add N per group
    for i, model in enumerate(['M1', 'M5', 'M6', 'M7']):
        n = len(df[df['model_type'] == model])
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={n}',
                ha='center', fontsize=8, color='gray')

    # Panel B: high-bias rate by model
    ax = axes[1]
    models = ['M1', 'M5', 'M6', 'M7']
    high_bias_rates = [df[df['model_type'] == m]['high_bias'].mean() * 100 for m in models]
    bars = ax.bar(range(len(models)), high_bias_rates, color=[MODEL_COLORS[m] for m in models],
                  alpha=0.7, width=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9, rotation=15)
    ax.set_ylabel('High-bias rate (% with |bias| > 5%)')
    ax.set_title('B: High-Bias Prevalence', fontweight='bold')
    ax.set_ylim(0, 105)

    # Add percentage labels
    for bar, rate in zip(bars, high_bias_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    _save_formats(fig, FIGURES_DIR / filename, tight=True)
    print(f"  [OK] Saved {filename}")
    plt.close(fig)


def fig_decision_tree(df, X, y, filename='fig_meta_decision_tree'):
    """Visualize the depth-3 decision tree as the diagnostic rule."""
    print(f"[4/4] Decision tree diagnostic rule ({filename})")

    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=SEED)
    dt.fit(X, y)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(dt, feature_names=FEATURES, filled=True,
              rounded=True, precision=1, fontsize=9,
              impurity=False, ax=ax)
    ax.set_title('Diagnostic Rule: Decision Tree for Predicting |Bias|', fontweight='bold')

    plt.tight_layout()
    _save_formats(fig, FIGURES_DIR / filename, tight=True)
    print(f"  [OK] Saved {filename}")
    plt.close(fig)


def run_all():
    """Generate all meta-model figures."""
    t0 = time.time()
    print("=" * 60)
    print("META-MODEL VISUALIZATIONS")
    print("=" * 60)

    df, X, y, y_high, df_te, X_tr, y_tr, X_te, y_te = load_and_prepare()

    # Quick sanity
    print(f"Features: {X.shape[1]}, Rows: {X.shape[0]}")
    print(f"Mean |bias|: {y.mean():.1f}%, High-bias rate: {y_high.mean()*100:.1f}%")

    # 1. Feature importance (trains RF)
    rf = fig_feature_importance(df, X, y)

    # 2. Predicted vs actual (held-out split)
    fig_predicted_vs_actual(df_te, X_tr, y_tr, X_te, y_te)

    # 3. Bias by model
    fig_bias_by_model(df)

    # 4. Decision tree
    fig_decision_tree(df, X, y)

    elapsed = time.time() - t0
    print(f"\nAll figures saved to {FIGURES_DIR}")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    run_all()
