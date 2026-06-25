# Experiment Execution Order

This document defines the canonical execution sequence for all thesis experiments,
aligned to **Chapter 3 (Methodology), Section 3.4 — Data Collection Procedure**.

All commands assume working directory: `/home/nisalperera/YOLOF/`

---

## Status Overview

| Stage | Thesis Step | Script | Status |
|---|---|---|---|
| Phase 1 | Step 1 — Reproducibility setup | *(manual)* | ✅ DONE |
| Phase 2a | Step 2a — Ingredient pool training (L1–L4, C1–C2) | `tools/train_net.py` | ✅ DONE |
| Step 2b | Step 2b — Pilot divergence check | `pipeline_runner.py --stage 2b` | ☐ |
| Step 3 | Step 3 — Ingredient diversity analysis | `ingredient_diversity_analysis.py` | ☐ |
| Step 4 | Step 4 — Ingredient pool review | *(manual review)* | ☐ |
| Step 5 | Step 5 — Ingredient quality audit | `ingredient_quality_audit.py` | ☐ |
| Step 6 | Step 6 — LMC barriers + Hessian traces | `lmc_hessian_analysis.py` | ☐ |
| Step 7 | Step 7 — Merging conditions C1–C6 + Greedy | `merge_conditions.py` + `merge_greedy.py` + `merge_tricomponent_greedy.py` | ☐ |
| Step 8 | Step 8 — Checkpoint pre-registration | `checkpoint_preregistration.py` | ☐ |
| Steps 9–10 | Steps 9–10 — Decoder fine-tuning D1, D2, C3 | `decoder_finetune.py` *(manual)* | ☐ |
| Step 11 | Step 11 — Hypothesis tests RQ1–RQ4 | `hypothesis_tests.py` | ☐ |
| Step 12 | Step 12 — COCO test-dev2017 final evaluation | `coco_testdev_eval.py` | ☐ |

---

## Quick Start — Full Automated Pipeline (DVC)

The preferred way to run all automated stages is via DVC, which handles dependency
tracking, caching, and reproducibility automatically:

```bash
cd /home/nisalperera/YOLOF

# Reproduce all pipeline stages from Step 2b onwards
dvc repro

# Reproduce a specific stage only
dvc repro merge_conditions
dvc repro hypothesis_tests
```

To run without DVC (manual orchestrator):

```bash
# Stages 2b → 8 (automated)
python -m yolof_soup.pipeline_runner --run-all

# Stages 11–12 after you complete Steps 9–10 manually
python -m yolof_soup.pipeline_runner --stage stats,finaleval --after-user-finetune
```

---

## Step-by-Step Execution

---

### Step 2b — Pilot Divergence Check

**Thesis reference:** Section 3.4, Step 2b  
**Purpose:** Confirm that all N=6 ingredient checkpoints (L1–L4, C1–C2) converged
and are numerically valid before committing to expensive analysis.

**Required inputs:**
- 6 checkpoints in `output/soup_exps/ingridients-refined/`

**Command:**
```bash
python -m yolof_soup.pipeline_runner --stage 2b
```

**Direct execution:**
```bash
python yolof_soup/experiments/ingredient_quality_audit.py \
  --phase2-output-dir output/soup_exps/ingridients-refined \
  --results-dir results \
  --outlier-threshold 3.0
```

**Outputs:**
- `results/ingredient_quality_audit_report.json`
- `results/ingredient_quality_audit_summary.txt`

**Expected runtime:** 15–30 min

---

### Step 3 — Ingredient Diversity Analysis

**Thesis reference:** Section 3.2.1 (hyperparameter-induced diversity criterion)  
**Purpose:** Measure weight-space diversity across the N=6 ingredient pool to confirm
the pool satisfies the diversity inclusion criterion for LMC analysis (MV1).

**Required inputs:**
- 6 ingredient checkpoints from Step 2b

**Command:**
```bash
python yolof_soup/experiments/ingredient_diversity_analysis.py \
  --checkpoint-dir output/soup_exps/ingridients-refined \
  --results-dir results
```

**Outputs:**
- `results/ingredient_diversity_report.json`
- `results/ingredient_diversity_summary.txt`

**Expected runtime:** 10–20 min

---

### Step 4 — Ingredient Pool Review *(manual)*

**Thesis reference:** Section 3.4, Step 4  
**Purpose:** Manually review Steps 2b and 3 outputs. Exclude any ingredient whose
mAP falls more than 3 pp below the pool maximum.

Confirm at least N=4 ingredients remain before proceeding.

---

### Step 5 — Ingredient Quality Audit

**Thesis reference:** Section 3.4, Step 5  
**Purpose:** Full COCO val2017 evaluation of all retained ingredients. Records per-class
AP values that feed directly into the RQ1 hypothesis test (H1).

**Command:**
```bash
python yolof_soup/experiments/ingredient_quality_audit.py \
  --phase2-output-dir output/soup_exps/ingridients-refined \
  --results-dir results \
  --outlier-threshold 3.0
```

**Outputs:**
- `results/ingredient_quality_audit_report.json`
- `results/ingredient_quality_audit_summary.txt`

**Expected runtime:** 15–30 min

---

### Step 6 — LMC Barriers + Hessian Traces (MV1, MV2)

**Thesis reference:** Section 3.5.2, RQ2/H2  
**Purpose:** Compute per-component loss landscape geometry to justify the
tri-component merging formulation (IV1):

- **MV1:** LMC barrier `B_c = max_α L_c((1−α)θA + αθB) − [L_c(θA) + L_c(θB)] / 2`
  over 15 model pairs × 5 components × 21-point α grid
- **MV2:** Hessian trace `Tr(H_c)` via Hutchinson estimator,
  50 Rademacher vectors, 6 checkpoints × 5 components

**Command:**
```bash
python yolof_soup/experiments/lmc_hessian_analysis.py --verbose
```

**Outputs:**
- `results/lmc_barriers.json` — Per-component barrier values (MV1)
- `results/hessian_traces.json` — Per-component Hessian traces (MV2)
- `results/lmc_hessian_report.txt` — Text summary

**Expected runtime:** 45–90 min

---

### Step 7 — Merging Conditions C1–C6 + Greedy Baselines

**Thesis reference:** Section 3.3.2  
**Purpose:** Construct and evaluate all soup variants. This step covers every
merging condition in the conceptual framework plus the greedy baselines:

| Script | Conditions | IV1 | IV2 |
|---|---|---|---|
| `merge_conditions.py` | C1–C6 | Absent/Present | Uniform/Dirichlet/Fisher/Learned |
| `merge_greedy.py` | Greedy (Wortsman 2022 baseline) | Global | — |
| `merge_tricomponent_greedy.py` | Tri-component greedy | Per-component | — |

**Commands:**
```bash
# All six merging conditions (C1–C6) — primary thesis experiment
python yolof_soup/experiments/merge_conditions.py --verbose

# Standard greedy soup baseline (Wortsman et al. 2022)
python yolof_soup/experiments/merge_greedy.py --verbose

# Tri-component greedy baseline
python yolof_soup/experiments/merge_tricomponent_greedy.py --verbose
```

**Or run via coefficient strategy comparison (C3–C6 head-to-head):**
```bash
python yolof_soup/experiments/merge_coefficient_strategies.py --verbose
```

**Outputs:**
- `results/merge_conditions_results.json` — mAP for C1–C6 on COCO val2017
- `results/merge_greedy_results.json` — Greedy soup mAP
- `results/merge_tricomponent_greedy_results.json` — Tri-component greedy mAP
- `results/merge_coefficient_strategies_results.json` — IV2 strategy comparison
- `checkpoints/soup_checkpoints/C1.pth` … `C6.pth` — Soup model checkpoints
- `checkpoints/soup_checkpoints/greedy.pth`
- `checkpoints/soup_checkpoints/tricomponent_greedy.pth`

**Expected runtime:** 60–180 min total (all variants)

---

### Step 8 — Checkpoint Pre-Registration

**Thesis reference:** Section 3.4, Step 8  
**Purpose:** Pre-register the source soup checkpoints for D1 and D2 **before**
any decoder-only fine-tuning begins. This prevents post-hoc selection bias (IV3).

**Required inputs:**
- `results/merge_conditions_results.json` from Step 7

**Command:**
```bash
python yolof_soup/experiments/checkpoint_preregistration.py \
  --soup-results results/merge_conditions_results.json \
  --output-dir results
```

**Outputs:**
- `results/checkpoint_preregistration.json` — Selected soup ID, SHA-256 hash, timestamp, rationale

**Expected runtime:** < 1 min

---

### Steps 9–10 — Decoder Fine-Tuning D1, D2, C3 *(manual)*

**Thesis reference:** Section 3.4, Steps 9–10  
**Purpose:** Fine-tune the decoder (cls_branch, reg_branch, object_pred only;
backbone + encoder frozen) initialised from three sources:

| Run ID | Source Checkpoint | GPU |
|---|---|---|
| D1 | Best of C1–C2 (component uniform soup) | RTX 5070 Ti |
| D2 | Best of C3–C6 (best learned soup, pre-registered in Step 8) | RTX 5070 Ti |
| C3 | Full three-stage pipeline (Step 7 + fine-tune) | RTX 5090 |

**Command:**
```bash
python yolof_soup/experiments/decoder_finetune.py \
  --run-id D1 \
  --source-checkpoint checkpoints/soup_checkpoints/C2.pth \
  --results-dir results

python yolof_soup/experiments/decoder_finetune.py \
  --run-id D2 \
  --source-checkpoint checkpoints/soup_checkpoints/<best_of_C3_C6>.pth \
  --results-dir results

python yolof_soup/experiments/decoder_finetune.py \
  --run-id C3 \
  --source-checkpoint checkpoints/soup_checkpoints/<best_of_C3_C6>.pth \
  --results-dir results
```

**Outputs:**
- `output/D1_checkpoint.pth`
- `output/D2_checkpoint.pth`
- `output/C3_checkpoint.pth`

**Expected runtime:** 2–4 hours total

---

### Step 11 — Hypothesis Tests RQ1–RQ4

**Thesis reference:** Section 3.5  
**Purpose:** Run all statistical tests to produce the Summary Decision Table (Table 3.3).

| Hypothesis | Test | Input |
|---|---|---|
| H1 — Soup > best individual (RQ1) | Paired t-test + Wilcoxon, 80 per-class APs | `merge_conditions_results.json` |
| H2 — Component structure matters (RQ2) | RM ANOVA + Tukey HSD on LMC barriers + Hessian traces | `lmc_barriers.json`, `hessian_traces.json` |
| H3 — IV2 coefficient strategy matters (RQ3) | One-way RM ANOVA across C3–C6 | `merge_coefficient_strategies_results.json` |
| H4 — Fine-tuning improves learned soup (RQ4) | Paired t-tests D1 vs D2, M5 vs M6 | `D1/D2/C3` results |

**Command:**
```bash
python yolof_soup/experiments/hypothesis_tests.py \
  --results-dir results \
  --output-dir results \
  --alpha 0.05
```

**Outputs:**
- `results/hypothesis_tests.json` — p-values, effect sizes, 95% CIs for all tests
- `results/hypothesis_tests_report.txt` — Formatted decision table
- `results/summary_decision_table.csv` — Table 3.3 (Hypothesis | Test | Result | Decision | Interpretation)

**Expected runtime:** 10–20 min

---

### Step 12 — COCO test-dev2017 Final Evaluation

**Thesis reference:** Section 3.4, Step 12  
**Purpose:** Submit the best overall model to the COCO evaluation server for the
unbiased headline mAP reported in Findings (Chapter 4). This is the only step
that touches the held-out test-dev2017 split.

**Models evaluated:** best of {C1–C6, greedy, tricomponent greedy, D1, D2, C3}

**Command:**
```bash
python yolof_soup/experiments/coco_testdev_eval.py \
  --checkpoint checkpoints/soup_checkpoints/<best_model>.pth \
  --output-dir results/coco_testdev
```

**Outputs:**
- `results/coco_testdev/detections_test-dev2017_<model>_results.json` — Submission file
- `results/coco_testdev/final_eval_report.txt` — mAP@0.5, mAP@0.5:0.95, per-class AP

**Expected runtime:** 30–60 min

---

## Smoke & Integration Checks

Run these at any time to verify the pipeline end-to-end on a mini-dataset
without requiring full GPU time:

```bash
# Lightweight smoke check (CPU-compatible, ~2 min)
python yolof_soup/experiments/smoke_integration.py

# Full integration run on 10-image COCO mini-subset
python yolof_soup/experiments/pipeline_integration.py --mini
```

---

## Full Manual Execution Sequence

```bash
cd /home/nisalperera/YOLOF

# Step 2b — Pilot divergence check
python yolof_soup/experiments/ingredient_quality_audit.py

# Step 3 — Diversity analysis
python yolof_soup/experiments/ingredient_diversity_analysis.py

# [Step 4 — Manual review]

# Step 5 — Full quality audit
python yolof_soup/experiments/ingredient_quality_audit.py

# Step 6 — LMC + Hessian
python yolof_soup/experiments/lmc_hessian_analysis.py --verbose

# Step 7 — All merging conditions
python yolof_soup/experiments/merge_conditions.py --verbose
python yolof_soup/experiments/merge_greedy.py --verbose
python yolof_soup/experiments/merge_tricomponent_greedy.py --verbose
python yolof_soup/experiments/merge_coefficient_strategies.py --verbose

# Step 8 — Pre-registration
python yolof_soup/experiments/checkpoint_preregistration.py

# [Steps 9–10 — Decoder fine-tuning D1, D2, C3 — manual]

# Step 11 — Hypothesis tests
python yolof_soup/experiments/hypothesis_tests.py

# Step 12 — Final COCO test-dev eval
python yolof_soup/experiments/coco_testdev_eval.py
```

---

## Monitoring

```bash
# Check running processes
ps aux | grep python.*yolof_soup

# Tail live logs
tail -f logs/experiment.log

# DVC pipeline status
dvc status

# Check disk usage
du -sh output/ results/ logs/
```

---

## Troubleshooting

### `Checkpoint not found`
```bash
ls -lh output/soup_exps/ingridients-refined/
# Expected: L1..L4, C1, C2 checkpoint files
```

### `CUDA out of memory`
```bash
export THESIS_BATCH_SIZE_PER_GPU=8
python -m yolof_soup.pipeline_runner --stage 7
```

### Re-run a failed stage
```bash
# Via DVC (recommended — uses caching)
dvc repro <stage_name>

# Via pipeline runner
python -m yolof_soup.pipeline_runner --stage 6
```

### Show current run configuration
```bash
python -m yolof_soup.pipeline_runner --show-config
```

---

## Environment Variables

```bash
export THESIS_ROOT=/home/nisalperera/YOLOF
export THESIS_NUM_GPUS=1
export THESIS_BATCH_SIZE_PER_GPU=16
export THESIS_DEVICE=cuda
export THESIS_N_INGREDIENTS=6
export COCO_ROOT=/path/to/coco
```

---

## Output Directory Structure

```
/home/nisalperera/YOLOF/
├── output/
│   └── soup_exps/ingridients-refined/   # Phase 2a ingredient checkpoints
│       ├── L1_checkpoint.pth
│       ├── L2_checkpoint.pth
│       ├── L3_checkpoint.pth
│       ├── L4_checkpoint.pth
│       ├── C1_checkpoint.pth
│       └── C2_checkpoint.pth
│
├── checkpoints/
│   └── soup_checkpoints/                # Step 7 soup model outputs
│       ├── C1.pth  (global uniform)
│       ├── C2.pth  (component uniform)
│       ├── C3.pth  (Dirichlet search)
│       ├── C4.pth  (Fisher-weighted)
│       ├── C5.pth  (Learned M5)
│       ├── C6.pth  (Learned M6)
│       ├── greedy.pth
│       └── tricomponent_greedy.pth
│
├── results/
│   ├── ingredient_quality_audit_report.json       # Step 5
│   ├── ingredient_diversity_report.json           # Step 3
│   ├── lmc_barriers.json                          # Step 6 (MV1)
│   ├── hessian_traces.json                        # Step 6 (MV2)
│   ├── merge_conditions_results.json              # Step 7 (C1–C6)
│   ├── merge_greedy_results.json                  # Step 7 (greedy)
│   ├── merge_tricomponent_greedy_results.json     # Step 7 (tri-component greedy)
│   ├── merge_coefficient_strategies_results.json  # Step 7 (IV2 comparison)
│   ├── checkpoint_preregistration.json            # Step 8
│   ├── hypothesis_tests.json                      # Step 11
│   ├── summary_decision_table.csv                 # Step 11 (Table 3.3)
│   └── coco_testdev/
│       └── final_eval_report.txt                  # Step 12
│
└── logs/
    └── experiment.log
```

---

## Timing Reference

| Step | Script | Runtime | Mode |
|---|---|---|---|
| 2b | `ingredient_quality_audit.py` | 15–30 min | Automated |
| 3 | `ingredient_diversity_analysis.py` | 10–20 min | Automated |
| 4 | *(manual review)* | — | Manual |
| 5 | `ingredient_quality_audit.py` | 15–30 min | Automated |
| 6 | `lmc_hessian_analysis.py` | 45–90 min | Automated |
| 7a | `merge_conditions.py` | 60–120 min | Automated |
| 7b | `merge_greedy.py` | 20–40 min | Automated |
| 7c | `merge_tricomponent_greedy.py` | 20–40 min | Automated |
| 7d | `merge_coefficient_strategies.py` | 30–60 min | Automated |
| 8 | `checkpoint_preregistration.py` | < 1 min | Automated |
| 9–10 | `decoder_finetune.py` (D1, D2, C3) | 2–4 hr | Manual |
| 11 | `hypothesis_tests.py` | 10–20 min | Automated |
| 12 | `coco_testdev_eval.py` | 30–60 min | Automated |

**Total automated time:** ~4–8 hours  
**Total manual time (Steps 9–10):** ~2–4 hours
