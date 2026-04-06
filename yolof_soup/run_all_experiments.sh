#!/usr/bin/env bash
# Master runner — executes all phases in order.
# Edit MODEL PATHS below to match your output/ directory.
# Usage: bash run_all_experiments.sh

set -euo pipefail

CFG="configs/yolof_R_50_DC5_1x_thesis_base.yaml"
BB_SOUP="output/soups/soup_backbone_encoder.pth"
OUT="output"

echo "========================================================="
echo "  Phase 3: Soup Construction"
echo "========================================================="
python tools/phase3_soup_construction.py \
  --config-file "$CFG" \
  --backbone-soup "$BB_SOUP" \
  --decoder-checkpoints \
      "$OUT/D1/model_best.pth" "$OUT/D2/model_best.pth" \
      "$OUT/D3/model_best.pth" "$OUT/D4/model_best.pth" \
      "$OUT/D5/model_best.pth" "$OUT/D6/model_best.pth" \
  --global-checkpoints \
      "$OUT/L1/model_best.pth" "$OUT/L2/model_best.pth" \
      "$OUT/L3/model_best.pth" "$OUT/L4/model_best.pth" \
  --baseline-checkpoint "$OUT/baseline/model_best.pth" \
  --output-dir "$OUT/phase3" \
  --max-eval-samples 1000

echo "========================================================="
echo "  Phase 4: Loss Landscape"
echo "========================================================="
python tools/phase4_loss_landscape.py \
  --config-file "$CFG" \
  --decoder-checkpoints \
      "$OUT/D1/model_best.pth" "$OUT/D2/model_best.pth" \
      "$OUT/D3/model_best.pth" "$OUT/D4/model_best.pth" \
      "$OUT/D5/model_best.pth" "$OUT/D6/model_best.pth" \
  --global-checkpoints \
      "$OUT/L1/model_best.pth" "$OUT/L2/model_best.pth" \
      "$OUT/L3/model_best.pth" "$OUT/L4/model_best.pth" \
  --phase3-results "$OUT/phase3/phase3_soup_results.json" \
  --output-dir "$OUT/phase4" \
  --num-alpha-samples 21 --max-eval-samples 500

echo "========================================================="
echo "  Phase 5: Cross-Domain (Pascal VOC 2007)"
echo "========================================================="
python tools/phase5_cross_domain.py \
  --config-file "$CFG" \
  --head-soup   "$OUT/phase3/learned_head_soup.pth" \
  --global-soup "$OUT/phase3/uniform_global_soup.pth" \
  --baseline    "$OUT/baseline/model_best.pth" \
  --voc-ann     "datasets/voc2007/annotations/instances_test2007.json" \
  --voc-img-dir "datasets/voc2007/images/test2007" \
  --output-dir  "$OUT/phase5"

echo "========================================================="
echo "  RQ1 Statistical Test"
echo "========================================================="
python tools/rq1_statistical_test.py \
  --config-file    "$CFG" \
  --phase3-results "$OUT/phase3/phase3_soup_results.json" \
  --output-dir     "$OUT/rq1"

echo "========================================================="
echo "  RQ3 Diversity Analysis"
echo "========================================================="
python tools/rq3_diversity_analysis.py \
  --config-file "$CFG" \
  --backbone-soup "$BB_SOUP" \
  --decoder-checkpoints \
      "$OUT/D1/model_best.pth" "$OUT/D2/model_best.pth" \
      "$OUT/D3/model_best.pth" "$OUT/D4/model_best.pth" \
      "$OUT/D5/model_best.pth" "$OUT/D6/model_best.pth" \
  --global-checkpoints \
      "$OUT/L1/model_best.pth" "$OUT/L2/model_best.pth" \
      "$OUT/L3/model_best.pth" "$OUT/L4/model_best.pth" \
      "$OUT/L5/model_best.pth" "$OUT/L6/model_best.pth" \
  --output-dir "$OUT/rq3"

echo "All experiments complete. Check output/ for JSON results."