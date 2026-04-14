#!/usr/bin/env bash
# Master runner for YOLOF-soup thesis experiments.
# Usage: bash yolof_soup/run_all_experiments.sh

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

run_step "Phase 3: Soup Construction" "yolof_soup.experiments.phase3_soup_construction"
run_step "Phase 4: Loss Landscape" "yolof_soup.experiments.phase4_loss_landscape"
run_step "Phase 5: Cross-Domain (Pascal VOC 2007)" "yolof_soup.experiments.phase5_cross_domain"
run_step "RQ1 Statistical Test" "yolof_soup.experiments.rq1_final_test"
run_step "RQ3 Diversity Analysis" "yolof_soup.experiments.rq3_diversity_analysis"

echo "All experiments complete. Check results and checkpoints directories."