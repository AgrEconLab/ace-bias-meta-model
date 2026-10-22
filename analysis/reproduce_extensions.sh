#!/usr/bin/env bash
# Revised extension analyses (run after reproduce_all.sh).
# Generates the configuration-level bias/MSE decomposition, threshold stability,
# negative-effect check, positivity diagnostics, correlated-covariate sensitivity,
# grouped feature importance/removal, target sensitivity, leave-one-DAG-out CV,
# training-range check, and classifier threshold sensitivity.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/../../logs"
mkdir -p "$LOGS"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0

SCRIPTS=(
    meta_config_bias.py
    meta_absolute_vs_relative.py
    meta_threshold_stability.py
    meta_negative_effect.py
    meta_positivity.py
    meta_multicov_correlated.py
    meta_group_ablation.py
    meta_m5_large_sample.py
    red_rule_2x2.py
    meta_target_sensitivity.py
    meta_lodo_cv.py
    meta_domain_check.py
    meta_classifier_threshold.py
)

SUMMARY="$LOGS/extensions_run_summary.tsv"
: > "$SUMMARY"
printf "script\trc\telapsed_s\tstart_utc\tend_utc\n" >> "$SUMMARY"

for script in "${SCRIPTS[@]}"; do
    name="${script%.py}"
    start_epoch=$(date +%s)
    start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "=== START $script $start_utc ===" | tee "$LOGS/ext_$name.log"
    ( cd "$HERE" && python -u "$script" ) 2>&1 | tee -a "$LOGS/ext_$name.log"
    rc=${PIPESTATUS[0]}
    end_epoch=$(date +%s)
    end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    elapsed=$((end_epoch - start_epoch))
    echo "=== END $script rc=$rc elapsed=${elapsed}s $end_utc ===" | tee -a "$LOGS/ext_$name.log"
    printf "%s\t%s\t%s\t%s\t%s\n" "$script" "$rc" "$elapsed" "$start_utc" "$end_utc" >> "$SUMMARY"
    if [ "$rc" -ne 0 ]; then
        echo "ABORT: $script failed"
        exit 1
    fi
done

echo "ALL EXTENSION SCRIPTS COMPLETED"
cat "$SUMMARY"
