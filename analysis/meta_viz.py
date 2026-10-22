"""
Meta-model: Diagnostic Visualizations

Generates the four manuscript figures:
  1. Predicted vs actual AREE scatter (RF, held-out M6/M7)
  2. Permutation feature importance bar chart
  3. AREE distribution and high-error prevalence by DAG
  4. Two-input decision tree (horizontal layout)

Figures are designed at approximately the 6-inch print width used in the
manuscript, so the point sizes below are close to the final printed sizes.
"""

import numpy as np
import pandas as pd
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from config import FIGURES_DIR

SEED = 20240625

FEATURES = [
    'n', 'y_mean', 'x_var',
    'logit_ace', 'logit_lpm_diff',
    'unadj_ace', 'adj_unadj_diff',
    'z_coef_mag', 'z_x_coef_mag', 'z_x_r2',
    'y_pred_min', 'y_pred_max', 'y_pred_range',
    'lpm_x_coef', 'y_model_r2',
]

# Print-size style: the manuscript embeds figures at 6.0 in width.
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
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

TREE_FEATURES = ['lpm_x_coef', 'n']

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


def fig_feature_importance(df, X_tr, y_tr, X_te, y_te, filename='fig_meta_importance'):
    """Bar chart of permutation feature importance from RF."""
    print(f"\n[1/4] Feature importance ({filename})")

    rf = RandomForestRegressor(n_estimators=1000, max_depth=10,
                               min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)

    from sklearn.inspection import permutation_importance
    imp = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_idx = np.argsort(imp.importances_mean)[::-1]

    fig, ax = plt.subplots(figsize=(6.9, 5.8))
    top_n = min(12, len(FEATURES))
    names = [FEATURES[i] for i in imp_idx[:top_n]]
    scores = imp.importances_mean[imp_idx[:top_n]]
    sds = imp.importances_std[imp_idx[:top_n]]

    # Short labels so the axis remains legible at the 6-inch print width.
    label_map = {
        'logit_ace': 'Logit ACE estimate',
        'lpm_ace': 'LPM ACE estimate',
        'lpm_x_coef': 'LPM X coefficient',
        'n': 'Sample size N',
        'unadj_ace': 'Unadjusted ACE',
        'logit_lpm_diff': '|Logit − LPM ACE|',
        'y_mean': 'Outcome mean',
        'y_pred_min': 'Min predicted P',
        'y_pred_range': 'Predicted P range',
        'y_pred_max': 'Max predicted P',
        'x_var': 'Var(X)',
        'z_coef_mag': '|Z| coefficient (logit)',
        'z_x_coef_mag': 'Z–X coefficient',
        'y_model_r2': 'Y model R-squared',
    }
    display_names = [label_map.get(n, n) for n in names]

    ax.barh(range(top_n), scores[::-1], xerr=sds[::-1], capsize=3,
            color='#2c3e50', alpha=0.88, height=0.62)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(display_names[::-1], fontsize=9)
    ax.set_xlabel('Permutation importance (AREE prediction)')
    ax.axvline(0, color='0.4', lw=0.7)
    xmin = min(0.0, float(scores.min()) * 1.25)
    xmax = float(scores.max()) * 1.15
    ax.set_xlim(xmin, xmax)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.18, linewidth=0.6)

    # Add R2 annotation
    r2 = r2_score(y_te, rf.predict(X_te))
    ax.text(0.97, 0.04, f'Held-out R² = {r2:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, fontstyle='italic')

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

    fig, ax = plt.subplots(figsize=(6.6, 5.0))

    for model in ['M6', 'M7']:
        mask = df_te['model_type'] == model
        ax.scatter(y_pred[mask], y_te[mask], c=MODEL_COLORS[model],
                   label=MODEL_LABELS[model], alpha=0.32, s=7, edgecolors='none')

    lim = max(y_te.max(), y_pred.max()) * 1.03
    ax.plot([0, lim], [0, lim], 'k--', lw=0.9, alpha=0.6, label='Perfect prediction')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Predicted AREE (%)')
    ax.set_ylabel('Actual AREE (%)')
    ax.legend(fontsize=8.5, markerscale=2.5, frameon=True,
              framealpha=0.9, edgecolor='0.8', loc='lower right')

    # R2 annotation (held-out)
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    ax.text(0.97, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f}pp',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9.5, fontstyle='italic',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    plt.tight_layout()
    _save_formats(fig, FIGURES_DIR / filename, tight=True)
    print(f"  [OK] Saved {filename}")
    plt.close(fig)


def fig_bias_by_model(df, filename='fig_meta_bias_by_model'):
    """Bias distribution by model family, highlighting structural patterns."""
    print(f"[3/4] Bias by model ({filename})")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.8),
                             gridspec_kw={'width_ratios': [1.2, 1.0]})

    # Single M-labels avoid overlap at the 6-inch print width; the caption
    # spells out the DAG names.
    short_labels = ['M1', 'M5', 'M6', 'M7']

    # Panel A: AREE distribution by model
    ax = axes[0]
    data_points = []
    colors = []
    for model in ['M1', 'M5', 'M6', 'M7']:
        vals = df[df['model_type'] == model]['abs_bias_pct'].values
        data_points.append(vals)
        colors.append(MODEL_COLORS[model])

    bp = ax.boxplot(data_points, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', lw=1.2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    boxprops=dict(lw=0.8), flierprops=dict(markersize=1.5,
                                                          markeredgewidth=0))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(short_labels, fontsize=9.5)
    ax.set_ylabel('AREE (%)')
    ax.set_title('A', loc='left', fontsize=11, fontweight='bold', pad=4)
    ax.set_yscale('symlog', linthresh=1)  # log-like but handles zeros
    ax.set_ylim(0, 2000)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.18, linewidth=0.6)

    # Panel B: high-error rate by model
    ax = axes[1]
    models = ['M1', 'M5', 'M6', 'M7']
    high_bias_rates = [df[df['model_type'] == m]['high_bias'].mean() * 100 for m in models]
    bars = ax.bar(range(len(models)), high_bias_rates, color=[MODEL_COLORS[m] for m in models],
                  alpha=0.7, width=0.55)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_labels, fontsize=9.5)
    ax.set_ylabel('High-error rate (%)')
    ax.set_title('B', loc='left', fontsize=11, fontweight='bold', pad=4)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.18, linewidth=0.6)

    # Add percentage labels
    for bar, rate in zip(bars, high_bias_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{rate:.0f}%', ha='center', fontsize=8.5, fontweight='bold')

    fig.subplots_adjust(wspace=0.35)
    plt.tight_layout()
    _save_formats(fig, FIGURES_DIR / filename, tight=True)
    print(f"  [OK] Saved {filename}")
    plt.close(fig)


def fig_decision_tree(df, X_tr, y_tr, filename='fig_meta_decision_tree'):
    """Visualize the two-input depth-3 tree as a horizontal diagnostic rule.

    ``sklearn.plot_tree`` squeezes eight leaf boxes into a 16-inch canvas; at
    the manuscript's 6-inch width the node text becomes unreadable.  This
    custom layout puts the three depth levels on the horizontal axis and the
    eight leaves on the vertical axis, so every threshold and leaf value stays
    legible in print.
    """
    print(f"[4/4] Decision tree diagnostic rule ({filename})")

    X_tr = X_tr[TREE_FEATURES]
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=SEED)
    dt.fit(X_tr, y_tr)
    tree = dt.tree_

    leaf_ids = [i for i in range(tree.node_count) if tree.children_left[i] == -1]
    leaf_y = {
        nid: 0.945 - k * (0.89 / (len(leaf_ids) - 1))
        for k, nid in enumerate(leaf_ids)
    }

    def ypos(nid):
        if nid in leaf_y:
            return leaf_y[nid]
        return (ypos(tree.children_left[nid]) + ypos(tree.children_right[nid])) / 2

    depth = {0: 0}

    def set_depth(nid, d):
        for child in (tree.children_left[nid], tree.children_right[nid]):
            if child != -1:
                depth[child] = d + 1
                set_depth(child, d + 1)

    set_depth(0, 0)
    xpos = {0: 0.12, 1: 0.375, 2: 0.635, 3: 0.88}

    zone_colors = {
        'RED': '#e74c3c',
        'YELLOW-A': '#f39c12',
        'YELLOW-B': '#f1c40f',
        'GREEN': '#2ecc71',
    }
    zone = {}

    def assign_zone(nid, current):
        if tree.children_left[nid] == -1:
            zone[nid] = current
            return
        if current == 'RED':
            assign_zone(tree.children_left[nid], 'RED')
            assign_zone(tree.children_right[nid], 'RED')
        elif depth[nid] == 0:
            assign_zone(tree.children_left[nid], 'RED')
            assign_zone(tree.children_right[nid], None)
        elif depth[nid] == 1:
            assign_zone(tree.children_left[nid], None)
            assign_zone(tree.children_right[nid], None)
        elif depth[nid] == 2:
            if tree.feature[nid] == 0:
                assign_zone(tree.children_left[nid], 'YELLOW-A')
                assign_zone(tree.children_right[nid], 'GREEN')
            else:
                assign_zone(tree.children_left[nid], 'YELLOW-B')
                assign_zone(tree.children_right[nid], 'YELLOW-A')

    assign_zone(0, None)

    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    box_w, box_h = 0.21, 0.08

    def draw_node(nid):
        x = xpos[depth[nid]]
        y = ypos(nid)
        is_leaf = tree.children_left[nid] == -1
        face = zone_colors[zone[nid]] if is_leaf else '#f2f2f2'
        alpha = 0.38 if is_leaf else 0.9
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.004,rounding_size=0.012',
            linewidth=0.8, edgecolor='#333333', facecolor=face,
            alpha=alpha, zorder=2,
        )
        ax.add_patch(box)

        if is_leaf:
            lines = [
                zone[nid],
                f'n = {tree.n_node_samples[nid]:,}',
                f'mean = {tree.value[nid][0][0]:.1f}',
            ]
        else:
            if tree.feature[nid] == 0:
                condition = f'LPM X ≤ {tree.threshold[nid]:.3f}'
            else:
                condition = f'n ≤ {tree.threshold[nid]:.0f}'
            lines = [
                condition,
                f'n = {tree.n_node_samples[nid]:,}',
                f'mean = {tree.value[nid][0][0]:.1f}',
            ]
        ax.text(x, y, '\n'.join(lines), ha='center', va='center',
                fontsize=8.5, linespacing=0.95, zorder=3)

        for child, label in ((tree.children_left[nid], 'Yes'),
                             (tree.children_right[nid], 'No')):
            if child == -1:
                continue
            x2 = xpos[depth[child]]
            y2 = ypos(child)
            arrow = FancyArrowPatch(
                (x + box_w / 2, y), (x2 - box_w / 2, y2),
                arrowstyle='-|>', mutation_scale=7, linewidth=0.8,
                color='#333333', zorder=1,
            )
            ax.add_patch(arrow)
            if depth[nid] == 0:
                ax.text((x + box_w / 2 + x2 - box_w / 2) / 2,
                        (y + y2) / 2 + 0.018, label,
                        ha='center', va='bottom', fontsize=8.0, color='#333333')
            draw_node(child)

    draw_node(0)
    ax.text(0.01, 1.0, 'Upper branch: condition true; lower branch: false.',
            ha='left', va='top', fontsize=7.5, color='0.35')

    plt.tight_layout(pad=0.15)
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
    rf = fig_feature_importance(df, X_tr, y_tr, X_te, y_te)

    # 2. Predicted vs actual (held-out split)
    fig_predicted_vs_actual(df_te, X_tr, y_tr, X_te, y_te)

    # 3. Bias by model
    fig_bias_by_model(df)

    # 4. Decision tree
    fig_decision_tree(df, X_tr, y_tr)

    elapsed = time.time() - t0
    print(f"\nAll figures saved to {FIGURES_DIR}")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    run_all()
