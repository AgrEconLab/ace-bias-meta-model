#!/usr/bin/env bash
# ==============================================================================
# reproduce_all.sh — Master reproduction script for the meta-model manuscript
#
# Usage:
#   bash analysis/reproduce_all.sh              # Full reproduction from scratch
#   bash analysis/reproduce_all.sh --quick       # Re-run only Phase 3 (tables + figures)
#   bash analysis/reproduce_all.sh --check       # Verify all output files exist
#
# Phases:
#   1. Generate raw data (long-running, ~40 min total)
#   2. Train and evaluate models (~25 min)
#   3. Export figures and tables (~2 min)
#
# Estimated total time (from scratch): ~60-70 minutes on a typical desktop.
# All scripts use fixed random seed 20240625 for deterministic results.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$PROJECT_DIR/results/raw"
FIGURES_DIR="$PROJECT_DIR/results/figures"
TABLES_DIR="$PROJECT_DIR/results/tables"

# ── Colors for output ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# ── Helper functions ──────────────────────────────────────────────────────────

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; }

section() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════════════"
}

timer_start() { TIMER=$SECONDS; }
timer_end()   {
    local elapsed=$(( SECONDS - TIMER ))
    local min=$(( elapsed / 60 ))
    local sec=$(( elapsed % 60 ))
    ok "Completed in ${min}m ${sec}s"
}

# ── File existence checks ───────────────────────────────────────────────────────

check_file() {
    if [ -f "$1" ]; then
        ok "Found: $(basename "$1")"
        return 0
    else
        warn "Missing: $(basename "$1")"
        return 1
    fi
}

# ── Phase 1: Generate raw data ───────────────────────────────────────────────

phase_1_generate_data() {
    section "PHASE 1/3: Generate raw data"

    # 1a. Meta-model training data (binary Y, 7 DAGs, 14,040 rows)
    info "meta_generate.py — training data (14,040 rows, ~10 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_generate.py
    timer_end
    check_file "$RAW_DIR/meta_training_data.parquet" || return 1

    # 1b. LPM vs Logit comparison (28 scenarios)
    info "run_lpm_compare.py — estimator parity analysis (~2 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/run_lpm_compare.py
    timer_end

    # 1c. Continuous outcome extension (14,040 rows)
    info "meta_linear.py — continuous Y data (~8 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_linear.py
    timer_end
    check_file "$RAW_DIR/meta_linear_data.parquet" || return 1

    # 1d. Multi-covariate extension (M8-M10)
    info "meta_multicov.py — multi-covariate data (~5 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_multicov.py
    timer_end

    # 1e. External validation (3 conditions, ~18 min combined)
    info "meta_external_val.py — external validation data (~18 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_external_val.py
    timer_end
    check_file "$RAW_DIR/meta_ext_1_Probit.parquet" || return 1
    check_file "$RAW_DIR/meta_ext_2_ExtendedRange.parquet" || return 1
    check_file "$RAW_DIR/meta_ext_3_UniformZ.parquet" || return 1

    echo ""
    ok "Phase 1 complete — all raw data generated."
}

# ── Phase 2: Train and evaluate models ───────────────────────────────────────

phase_2_train_evaluate() {
    section "PHASE 2/3: Train and evaluate models"

    # 2a. Meta-model training (RF, decision tree, logistic regression)
    info "meta_train.py — train RF + decision tree + logistic (~1 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_train.py
    timer_end

    # 2b. Robustness (bootstrap CI + hyperparameter sensitivity, ~7 min)
    info "meta_robustness.py — bootstrap CI + sensitivity (~7 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_robustness.py
    timer_end
    check_file "$RAW_DIR/meta_robustness_results.json" || return 1

    # 2c. Extended bootstrap (binary + continuous Y, ~8 min)
    info "meta_bootstrap_extended.py — extended bootstrap CIs (~8 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_bootstrap_extended.py
    timer_end
    check_file "$RAW_DIR/meta_bootstrap_extended_results.json" || return 1

    # 2d. External validation bootstrap (~12 min)
    info "meta_bootstrap_external.py — external validation bootstrap (~12 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_bootstrap_external.py
    timer_end
    check_file "$RAW_DIR/meta_bootstrap_external_results.json" || return 1

    # 2e. Diagnostic: remove logit_ace robustness check
    info "meta_no_logit_ace.py — diagnostic robustness check (~1 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_no_logit_ace.py
    timer_end

    echo ""
    ok "Phase 2 complete — all models trained and evaluated."
}

# ── Phase 3: Export figures and tables ──────────────────────────────────────

phase_3_figures_tables() {
    section "PHASE 3/3: Export figures and tables"

    # 3a. Figures (4 panels, ~1 min)
    info "meta_viz.py — generate 4 manuscript figures (~1 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_viz.py
    timer_end

    # Verify all 4 figures
    local figures_ok=0
    for fig in fig_meta_importance fig_meta_pred_vs_actual fig_meta_bias_by_model fig_meta_decision_tree; do
        if check_file "$FIGURES_DIR/${fig}.pdf"; then
            figures_ok=$((figures_ok + 1))
        fi
    done
    if [ "$figures_ok" -eq 4 ]; then
        ok "All 4 figures present in $FIGURES_DIR"
    else
        warn "Expected 4 figures, found $figures_ok"
    fi

    # 3b. Tables 1-3 (feature importance, per-DAG R², decision rule)
    info "meta_export_tables.py — export Tables 1-3 (~1 min)..."
    timer_start
    cd "$PROJECT_DIR" && python analysis/meta_export_tables.py
    timer_end

    # Verify tables
    local tables_ok=0
    for tbl in table1_feature_importance table2_per_dag_rsquared table3_decision_rule; do
        if check_file "$TABLES_DIR/${tbl}.csv"; then
            tables_ok=$((tables_ok + 1))
        fi
    done
    if [ "$tables_ok" -eq 3 ]; then
        ok "All 3 manuscript tables present in $TABLES_DIR"
    else
        warn "Expected 3 tables, found $tables_ok"
    fi

    echo ""
    ok "Phase 3 complete — all figures and tables exported."
}

# ── Verification mode ─────────────────────────────────────────────────────────

check_all_outputs() {
    section "CHECK: Verify all output files exist"

    local missing=0

    # Raw data files
    info "Checking raw data files..."
    check_file "$RAW_DIR/meta_training_data.parquet"        || missing=$((missing + 1))
    check_file "$RAW_DIR/meta_linear_data.parquet"          || missing=$((missing + 1))
    check_file "$RAW_DIR/meta_ext_1_Probit.parquet"         || missing=$((missing + 1))
    check_file "$RAW_DIR/meta_ext_2_ExtendedRange.parquet"  || missing=$((missing + 1))
    check_file "$RAW_DIR/meta_ext_3_UniformZ.parquet"       || missing=$((missing + 1))

    # JSON results
    info "Checking JSON results..."
    check_file "$RAW_DIR/meta_robustness_results.json"      || missing=$((missing + 1))
    check_file "$RAW_DIR/meta_bootstrap_extended_results.json" || missing=$((missing + 1))
    check_file "$RAW_DIR/meta_bootstrap_external_results.json" || missing=$((missing + 1))

    # Figures
    info "Checking figures..."
    check_file "$FIGURES_DIR/fig_meta_importance.pdf"       || missing=$((missing + 1))
    check_file "$FIGURES_DIR/fig_meta_pred_vs_actual.pdf"   || missing=$((missing + 1))
    check_file "$FIGURES_DIR/fig_meta_bias_by_model.pdf"    || missing=$((missing + 1))
    check_file "$FIGURES_DIR/fig_meta_decision_tree.pdf"    || missing=$((missing + 1))

    # Tables
    info "Checking tables..."
    check_file "$TABLES_DIR/table1_feature_importance.csv"  || missing=$((missing + 1))
    check_file "$TABLES_DIR/table2_per_dag_rsquared.csv"    || missing=$((missing + 1))
    check_file "$TABLES_DIR/table3_decision_rule.csv"       || missing=$((missing + 1))
    check_file "$TABLES_DIR/table3_decision_tree_text.txt"  || missing=$((missing + 1))
    check_file "$TABLES_DIR/lpm_logit_comparison.csv"       || missing=$((missing + 1))
    for fig in fig_meta_importance fig_meta_pred_vs_actual fig_meta_bias_by_model fig_meta_decision_tree; do
        check_file "$FIGURES_DIR/${fig}.jpg" || missing=$((missing + 1))
    done

    echo ""
    if [ "$missing" -eq 0 ]; then
        ok "All expected output files present."
    else
        warn "$missing output file(s) missing. Re-run the corresponding phase."
        return 1
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    local mode="${1:-}"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║      Meta-Model Manuscript — Full Reproduction Pipeline        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo "  Project: $PROJECT_DIR"
    echo "  Mode:    ${mode:-full}"
    echo "  Date:    $(date)"
    echo ""

    case "$mode" in
        "--check")
            check_all_outputs
            ;;
        "--quick")
            phase_3_figures_tables
            echo ""
            ok "Quick reproduction complete (Phase 3 only)."
            ;;
        "--full"|"")
            phase_1_generate_data
            phase_2_train_evaluate
            phase_3_figures_tables
            echo ""
            ok "Full reproduction complete — all results regenerated from scratch."
            echo ""
            echo "  Summary:"
            echo "    Raw data:   $RAW_DIR/"
            echo "    Figures:    $FIGURES_DIR/  (4 PDF + 4 JPG files)"
            echo "    Tables:     $TABLES_DIR/   (4 CSV files)"
            echo "    JSON:       $RAW_DIR/      (3 result files)"
            echo ""
            check_all_outputs
            ;;
        *)
            echo "Usage: bash analysis/reproduce_all.sh [--quick | --check]"
            echo ""
            echo "  (no flag)    Full reproduction: Phase 1 → 2 → 3 (~60-70 min)"
            echo "  --quick      Phase 3 only: figures + tables from existing data (~2 min)"
            echo "  --check      Verify all output files exist"
            exit 1
            ;;
    esac
}

main "$@"
