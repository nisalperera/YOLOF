#!/usr/bin/env bash
# Master runner for YOLOF-soup thesis experiments.
# Usage: bash yolof_soup/run_all_experiments.sh
# Or:   bash yolof_soup/run_all_experiments.sh --integration (full integration test with validation)

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

run_step() {
  local title="$1"
  local module="$2"

  echo "========================================================="
  echo "  ${title}"
  echo "========================================================="
  "${PYTHON_BIN}" -m "${module}"
}

# Check if integration mode is requested
if [[ "${1:-}" == "--integration" ]]; then
  echo "========================================================="
  echo "  RUNNING FULL INTEGRATION TEST"
  echo "========================================================="
  "${PYTHON_BIN}" -m yolof_soup.experiments.integration_test
  exit $?
fi

# Standard mode: run all phases
run_step "Phase 3: Soup Construction (M1-M4)" "yolof_soup.experiments.phase3_soup_construction"
run_step "Phase 4: Loss Landscape (LMC + Hessian)" "yolof_soup.experiments.phase4_loss_landscape"
run_step "Phase 5: Head Fine-Tuning (D1/D2/C3)" "yolof_soup.experiments.phase5_head_finetuning"
run_step "RQ3: Coefficient Strategy Tests" "yolof_soup.experiments.rq3_coefficient_strategy_test"
run_step "RQ4: Full Pipeline & Per-Category Analysis" "yolof_soup.experiments.rq4_full_pipeline_test"

echo "========================================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "========================================================="
echo "Results saved to: $(pwd)/results"
echo ""
echo "Run integration test for full validation:"
echo "  bash yolof_soup/run_all_experiments.sh --integration"