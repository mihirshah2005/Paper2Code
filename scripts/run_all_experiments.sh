#!/bin/bash
# run_all_experiments.sh
#
# Runs all 5 experiment configurations for a single paper, then evaluates each.
# Assumes you have already run 1_planning.py for this paper.
#
# Usage:
#   bash run_all_experiments.sh <paper_name> <pdf_json_path> <base_output_dir> <data_dir>
#
# Example:
#   bash run_all_experiments.sh transformer data/transformer.json outputs/transformer ../data

set -e

PAPER_NAME=$1
PDF_JSON_PATH=$2
BASE_OUTPUT_DIR=$3
DATA_DIR=${4:-"../data"}
GPT_VERSION=${5:-"o3-mini"}
PAPER_FORMAT=${6:-"JSON"}

if [ -z "$PAPER_NAME" ] || [ -z "$PDF_JSON_PATH" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Usage: bash run_all_experiments.sh <paper_name> <pdf_json_path> <base_output_dir> [data_dir] [gpt_version] [paper_format]"
    exit 1
fi

echo "============================================"
echo "Running all experiments for: $PAPER_NAME"
echo "============================================"

# The planning output should already exist at BASE_OUTPUT_DIR
# We create separate sub-dirs for each experiment's analyzing artifacts

run_experiment() {
    local MODE=$1
    local FORMAT=$2
    local EXP_NAME="${MODE}_${FORMAT}"
    local EXP_OUTPUT="${BASE_OUTPUT_DIR}/experiments/${EXP_NAME}"

    echo ""
    echo "--------------------------------------------"
    echo "Experiment: $EXP_NAME"
    echo "--------------------------------------------"

    # Create experiment output dir and symlink planning artifacts
    mkdir -p "$EXP_OUTPUT"

    # Copy/link planning artifacts so the experiment can find them
    if [ ! -f "$EXP_OUTPUT/planning_trajectories.json" ]; then
        cp "$BASE_OUTPUT_DIR/planning_trajectories.json" "$EXP_OUTPUT/" 2>/dev/null || true
        cp "$BASE_OUTPUT_DIR/planning_config.yaml" "$EXP_OUTPUT/" 2>/dev/null || true
        cp "$BASE_OUTPUT_DIR/task_list.json" "$EXP_OUTPUT/" 2>/dev/null || true
    fi

    # Determine repo dir (for static analysis on existing code)
    REPO_DIR="${BASE_OUTPUT_DIR}/${PAPER_NAME}_repo"

    # Run the analyzing experiment
    python 2_analyzing_experiments.py \
        --paper_name "$PAPER_NAME" \
        --gpt_version "$GPT_VERSION" \
        --paper_format "$PAPER_FORMAT" \
        --pdf_json_path "$PDF_JSON_PATH" \
        --output_dir "$EXP_OUTPUT" \
        --mode "$MODE" \
        --feedback_format "$FORMAT" \
        --output_repo_dir "$REPO_DIR"

    echo "Experiment $EXP_NAME complete."
    echo "Token summary: $(cat $EXP_OUTPUT/experiment_summary_${MODE}_${FORMAT}.json | python -c 'import sys,json; d=json.load(sys.stdin); print(f"total={d[\"total_tokens\"]:,}")')"
}


# =====================================================
# EXPERIMENT 1: LLM-only with structured JSON (baseline loop)
# This is your original 2_analyzing.py behavior
# =====================================================
run_experiment "llm_only" "json"

# =====================================================
# EXPERIMENT 2: LLM-only with free-text (ablation)
# Same rubric, but no JSON schema constraint
# =====================================================
run_experiment "llm_only" "freetext"

# =====================================================
# EXPERIMENT 3: Static-only (no LLM judge at all)
# Only ast.parse + pylint + import probe
# =====================================================
run_experiment "static_only" "json"

# =====================================================
# EXPERIMENT 4: Multi-signal (all channels combined)
# LLM judge + static analysis + execution probes
# =====================================================
run_experiment "multi_signal" "json"


echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE FOR: $PAPER_NAME"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Run 3_coding.py for each experiment's output to generate repos"
echo "2. Run eval.py on each repo to get rubric scores"
echo "3. Run collect_results.py to gather all scores into a table"
echo ""
echo "Example for one experiment:"
echo "  python 3_coding.py --paper_name $PAPER_NAME --output_dir $BASE_OUTPUT_DIR/experiments/llm_only_json --output_repo_dir $BASE_OUTPUT_DIR/experiments/llm_only_json/${PAPER_NAME}_repo ..."
echo "  python eval.py --paper_name $PAPER_NAME --target_repo_dir $BASE_OUTPUT_DIR/experiments/llm_only_json/${PAPER_NAME}_repo ..."