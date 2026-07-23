"""
LPM vs Logit: Systematic ACE Bias Comparison

Compares Linear Probability Model (OLS) vs Logit g-computation
for ACE estimation across all 7 DAG models × 4 cases.

Key question: For which scenarios does LPM give less/more biased ACE estimates?
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgp import simulate_model, get_true_ace, get_true_ace_case
from estimators import logit_ace_gcomp, lpm_ace_ols, probit_ace_gcomp
from config import CONFIG
import warnings
warnings.filterwarnings('ignore')

SEED = 20240625
N_ITER = 200
N_SAMPLE = 1000


def run_comparison(n_iter=N_ITER, n_sample=N_SAMPLE):
    print("=" * 70)
    print("LPM vs LOGIT: Systematic ACE Bias Comparison")
    print("=" * 70)

    models = CONFIG.dag_models  # M1-M7
    cases = CONFIG.cases        # a-d

    all_rows = []

    for model in models:
        for case in cases:
            # Use case-specific true ACE
            true_ace = get_true_ace_case(model, case, n_mc=200000, seed=42)

            logit_aces = np.zeros(n_iter)
            lpm_aces = np.zeros(n_iter)
            probit_aces = np.zeros(n_iter)

            for it in range(n_iter):
                data = simulate_model(model, case, n=n_sample, seed=SEED + it)

                logit_r = logit_ace_gcomp(data, bootstrap=False)
                lpm_r = lpm_ace_ols(data, bootstrap=False)
                probit_r = probit_ace_gcomp(data, bootstrap=False)

                logit_aces[it] = logit_r['ace']
                lpm_aces[it] = lpm_r['ace']
                probit_aces[it] = probit_r['ace']

            logit_m = float(np.mean(logit_aces))
            lpm_m = float(np.mean(lpm_aces))
            probit_m = float(np.mean(probit_aces))

            logit_s = float(np.std(logit_aces, ddof=1))
            lpm_s = float(np.std(lpm_aces, ddof=1))
            probit_s = float(np.std(probit_aces, ddof=1))

            logit_b = (logit_m - true_ace) / abs(true_ace) * 100
            lpm_b = (lpm_m - true_ace) / abs(true_ace) * 100
            probit_b = (probit_m - true_ace) / abs(true_ace) * 100

            # Improvement: positive = LPM reduces |bias| relative to logit
            logit_abs = abs(logit_b)
            lpm_abs = abs(lpm_b)
            impr = (logit_abs - lpm_abs) / logit_abs * 100 if logit_abs > 1e-10 else 0

            scenario = f"{model}-{case}"
            print(f"  {scenario:<8} true={true_ace:.5f}  "
                  f"logit={logit_m:.5f}({logit_b:+.1f}%)  "
                  f"LPM={lpm_m:.5f}({lpm_b:+.1f}%)  "
                  f"|LPM|={lpm_abs:.1f}% vs |logit|={logit_abs:.1f}%  "
                  f"{'LPM wins!' if impr > 5 else ('Logit wins!' if impr < -5 else 'tie')}")

            all_rows.append({
                'model': model, 'case': case, 'scenario': scenario,
                'true_ace': true_ace,
                'logit_ace': logit_m, 'logit_bias_pct': round(logit_b, 3), 'logit_sd': logit_s,
                'lpm_ace': lpm_m, 'lpm_bias_pct': round(lpm_b, 3), 'lpm_sd': lpm_s,
                'probit_ace': probit_m, 'probit_bias_pct': round(probit_b, 3), 'probit_sd': probit_s,
                'improvement_pct': round(impr, 1),
                'lpm_wins': impr > 5,
                'logit_wins': impr < -5,
            })

    # ── Summary ──
    df = pd.DataFrame(all_rows)
    print(f"\n{'=' * 70}")
    print("SUMMARY: Bias% by Scenario (Logit vs LPM vs Probit)")
    print(f"{'=' * 70}")
    cols = ['logit_bias_pct', 'lpm_bias_pct', 'probit_bias_pct', 'improvement_pct']
    for c in cols:
        df[c] = df[c].astype(float)

    print(f"\n{'Scenario':<10} {'Logit%':>9} {'LPM%':>9} {'Probit%':>9} {'Improve':>9}  {'Winner':<12}")
    print(f"{'─'*10} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*12}")
    for _, r in df.iterrows():
        winner = 'LPM' if r['lpm_wins'] else ('Logit' if r['logit_wins'] else 'tie')
        print(f"{r['scenario']:<10} {r['logit_bias_pct']:>+8.1f}% {r['lpm_bias_pct']:>+8.1f}% "
              f"{r['probit_bias_pct']:>+8.1f}% {r['improvement_pct']:>+8.0f}%  {winner:<12}")

    # Win/loss count
    n_lpm = df['lpm_wins'].sum()
    n_logit = df['logit_wins'].sum()
    n_tie = len(df) - n_lpm - n_logit
    print(f"\n  Summary: LPM wins {n_lpm}/{len(df)}, "
          f"Logit wins {n_logit}/{len(df)}, tie {n_tie}/{len(df)}")

    return df


if __name__ == '__main__':
    result = run_comparison()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../results/tables/lpm_logit_comparison.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\n[OK] Saved {out}")
