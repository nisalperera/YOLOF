# Experiment Execution Order & Individual Commands

This document provides the exact command sequence to run all experiments individually from Phase 2b through Phase 7.

---

## Overview

**Project Status:**
- ✅ Phase 1: Reproducibility setup & pilot check (COMPLETED)
- ✅ Phase 2a: Ingredient pool training (L1-L4, C1-C2) (COMPLETED)
- ☐ Phase 2b-7: Automated analysis (THIS DOCUMENT)
- ☐ Phase 5a-5b: User-managed fine-tuning (MANUAL)

**Working Directory:** `/home/nisalperera/YOLOF/`

All commands below assume you are in this directory.

---

## Quick Start: Run All Automated Phases Sequentially

To run Phases 2b → 4b (takes ~1-2 hours):

```bash
cd /home/nisalperera/YOLOF
python yolof_soup/run_phases_2b_to_7.py --run-all
```

After you complete Phase 5a and 5b manually, run Phases 6-7:

```bash
python yolof_soup/run_phases_2b_to_7.py --phase 6,7 --after-user-phase-5
```

---

## Phase-by-Phase Individual Commands

### **Phase 2b: Ingredient Quality Audit**

**Purpose:** Evaluate 6 Phase 2a ingredients (L1-L4, C1-C2) on COCO val2017, flag outliers.

**Required Inputs:**
- 6 checkpoints in `checkpoints/output/`: `L1_checkpoint.pth`, `L2_checkpoint.pth`, ..., `C2_checkpoint.pth`

**Command (via Master Orchestrator):**
```bash
python yolof_soup/run_phases_2b_to_7.py --phase 2b
```

**Command (Direct Execution):**
```bash
python yolof_soup/experiments/quality_audit.py \
  --phase2-output-dir checkpoints/output \
  --results-dir results \
  --outlier-threshold 3.0
```

**Output Files:**
- `results/phase2b_audit_report.json` – Detailed per-ingredient metrics
- `results/phase2b_audit_summary.txt` – Human-readable summary

**Expected Runtime:** 15–30 minutes

**Options:**
- `--phase2-output-dir`: Override Phase 2a checkpoint location (default: `checkpoints/output`)
- `--results-dir`: Override results output directory (default: `results`)
- `--outlier-threshold`: mAP threshold (pp) for flagging outliers (default: 3.0)

---

### **Phase 3: Loss Landscape Measurement**

**Purpose:** Compute linear mode connectivity (LMC) barriers and Hessian traces between ingredient pairs.

**Required Inputs:**
- 6 ingredient checkpoints from Phase 2b audit

**Command (via Master Orchestrator):**
```bash
python yolof_soup/run_phases_2b_to_7.py --phase 3
```

**Command (Direct Execution):**
```bash
python yolof_soup/experiments/loss_landscape.py --verbose
```

**Output Files:**
- `results/phase3_lmc_results.json` – Barrier measurements (4-point grids between pairs)
- `results/phase3_hessian_traces.json` – Hessian trace estimates
- `results/phase3_landscape_report.txt` – Text summary

**Expected Runtime:** 45–90 minutes

**Options:**
- `--verbose`: Enable DEBUG-level logging

---

### **Phase 4: Soup Construction & Evaluation**

**Purpose:** Construct 4 soup variants (M1, M2, M3, M4) and evaluate on COCO val2017.

**Required Inputs:**
- Phase 3 loss landscape results (used for M4 Fisher weighting)
- 6 ingredient checkpoints

**Command (via Master Orchestrator):**
```bash
python yolof_soup/run_phases_2b_to_7.py --phase 4
```

**Command (Direct Execution):**
```bash
python yolof_soup/experiments/soup_construction.py --verbose
```

**Output Files:**
- `results/phase4_soup_results.json` – Metrics for all 4 soups (M1-M4)
- `checkpoints/soup_checkpoints/M1.pth`, `M2.pth`, `M3.pth`, `M4.pth` – Soup model checkpoints
- `results/phase4_soup_report.txt` – Performance comparison

**Expected Runtime:** 60–120 minutes

**Options:**
- `--verbose`: Enable DEBUG-level logging

---

### **Phase 4b: Pre-Registration of Best Learned Soup**

**Purpose:** Pre-register the best soup (M1-M4) to prevent post-hoc selection bias before Phase 5.

**Required Inputs:**
- `results/phase4_soup_results.json` from Phase 4

**Command (via Master Orchestrator):**
```bash
python yolof_soup/run_phases_2b_to_7.py --phase 4b
```

**Command (Direct Execution):**
```bash
python yolof_soup/experiments/preregistration.py \
  --soup-results results/phase4_soup_results.json \
  --output-dir results
```

**Output Files:**
- `results/phase4b_preregistration.json` – Selected soup ID, timestamp, rationale

**Expected Runtime:** < 1 minute

**Options:**
- `--soup-results`: Path to Phase 4 soup results (default: `results/phase4_soup_results.json`)
- `--output-dir`: Results output directory (default: `results`)

---

### **Phase 5a & 5b: Decoder Fine-Tuning (USER-MANAGED)**

**⚠️ You run these manually.** See [../YOLOF_SOUP_QUICK_REFERENCE.md](../YOLOF_SOUP_QUICK_REFERENCE.md#phase-5-decoder-fine-tuning) for instructions.

**What Happens:**
1. **Phase 5a:** Fine-tune two best single ingredients (heads only) → `D1_checkpoint.pth`, `D2_checkpoint.pth`
2. **Phase 5b:** Fine-tune best learned soup (heads only) → `C3_checkpoint.pth`

**Inputs Needed:**
- Best ingredient from Phase 2b audit
- Second-best ingredient from Phase 2b
- Best learned soup from Phase 4b pre-registration

**Expected Runtime:** 2–4 hours total

---

### **Phase 6: Archive & Experiment Summary**

**Purpose:** Archive all results, compute file hashes, generate final experiment summary.

**Required Inputs:**
- All outputs from Phases 2b, 3, 4, 4b, 5a, 5b

**Command (via Master Orchestrator):**
```bash
python yolof_soup/run_phases_2b_to_7.py --phase 6 --after-user-phase-5
```

**Command (Direct Execution):**
```bash
python yolof_soup/experiments/archive_and_summary.py \
  --results-dir results \
  --output-dir results/archive
```

**Output Files:**
- `results/experiment_summary.json` – Complete experiment metadata + SHA-256 hashes
- `results/experiment_lineage.txt` – Data provenance and dependencies
- `results/archive/` – Compressed archive of all outputs

**Expected Runtime:** 5–10 minutes

**Options:**
- `--results-dir`: Results directory to archive (default: `results`)
- `--output-dir`: Archive output directory (default: `results/archive`)
- `--compress`: Create .tar.gz archive (boolean flag)

---

### **Phase 7: Statistical Analysis & Hypothesis Testing**

**Purpose:** Run 12 hypothesis tests across RQ1-RQ4 to validate model soup benefits.

**Required Inputs:**
- All outputs from Phases 2b-6

**Command (via Master Orchestrator):**
```bash
python yolof_soup/run_phases_2b_to_7.py --phase 7 --after-user-phase-5
```

**Command (Direct Execution):**
```bash
python yolof_soup/experiments/statistical_analysis.py \
  --results-dir results \
  --output-dir results
```

**Output Files:**
- `results/phase7_hypothesis_tests.json` – Test results: p-values, effect sizes, 95% CIs
- `results/phase7_statistical_report.txt` – Formatted test report with conclusions

**Expected Runtime:** 10–20 minutes

**Options:**
- `--results-dir`: Results directory (default: `results`)
- `--output-dir`: Output directory (default: `results`)
- `--alpha`: Significance level (default: 0.05)

---

## Complete Execution Sequence

### **Option A: Fully Automated (You wait for user Phase 5)**

```bash
# Terminal 1: Run phases 2b-4b automatically
cd /home/nisalperera/YOLOF
python yolof_soup/run_phases_2b_to_7.py --run-all

# Expected time: ~1-2 hours
# Outputs: Phases 2b, 3, 4, 4b results

# [YOU COMPLETE PHASES 5a & 5b MANUALLY HERE]

# Terminal 1: Resume with phases 6-7
python yolof_soup/run_phases_2b_to_7.py --phase 6,7 --after-user-phase-5

# Expected time: ~30 minutes
# Outputs: Archive and statistical analysis
```

### **Option B: Manual Phase-by-Phase**

```bash
cd /home/nisalperera/YOLOF

# Phase 2b: Audit ingredients
python yolof_soup/experiments/quality_audit.py

# Phase 3: Measure loss landscape
python yolof_soup/experiments/loss_landscape.py --verbose

# Phase 4: Construct soups
python yolof_soup/experiments/soup_construction.py --verbose

# Phase 4b: Pre-register best soup
python yolof_soup/experiments/preregistration.py

# [USER RUNS 5a & 5b]

# Phase 6: Archive results
python yolof_soup/experiments/archive_and_summary.py

# Phase 7: Statistical analysis
python yolof_soup/experiments/statistical_analysis.py
```

---

## Monitoring Execution

### **Check if phases are running:**
```bash
ps aux | grep python.*yolof_soup
```

### **View live logs:**
```bash
tail -f logs/phase_2b.log          # Phase 2b logs
tail -f logs/phase_3.log           # Phase 3 logs
# ... and so on
```

### **Check disk space (checkpoints + results are large):**
```bash
du -sh checkpoints/ results/
```

---

## Troubleshooting

### **Phase 2b fails: "Checkpoint not found"**
- Verify: `ls -lh checkpoints/output/`
- Should show exactly 6 files: `L1_checkpoint.pth` through `C2_checkpoint.pth`
- See [../PHASE_2A_CHECKPOINT_STRUCTURE.md](../PHASE_2A_CHECKPOINT_STRUCTURE.md)

### **Any phase fails: "CUDA out of memory"**
```bash
export THESIS_BATCH_SIZE_PER_GPU=8
python yolof_soup/run_phases_2b_to_7.py --phase 2b
```

### **Run single phase again after failure:**
```bash
# All phases create output files that can be re-used
# Just re-run the failed phase:
python yolof_soup/run_phases_2b_to_7.py --phase 3
```

### **View detailed config:**
```bash
python yolof_soup/run_phases_2b_to_7.py --show-config
```

---

## Environment Variables

Override defaults by setting before running:

```bash
# Override batch size per GPU
export THESIS_BATCH_SIZE_PER_GPU=8

# Override GPU count
export THESIS_NUM_GPUS=1

# Override project root
export THESIS_ROOT=/path/to/project

# Override compute device
export THESIS_DEVICE=cuda

# Then run phases:
python yolof_soup/run_phases_2b_to_7.py --phase 2b
```

---

## Output Directory Structure

After all phases complete:

```
/home/nisalperera/YOLOF/results/
├── phase2b_audit_report.json
├── phase2b_audit_summary.txt
├── phase3_lmc_results.json
├── phase3_hessian_traces.json
├── phase3_landscape_report.txt
├── phase4_soup_results.json
├── phase4_soup_report.txt
├── phase4b_preregistration.json
├── phase6_experiment_summary.json
├── phase6_experiment_lineage.txt
├── phase7_hypothesis_tests.json
└── phase7_statistical_report.txt

/home/nisalperera/YOLOF/checkpoints/
├── output/                    (Phase 2a ingredients)
│   ├── L1_checkpoint.pth
│   ├── L2_checkpoint.pth
│   ...
└── soup_checkpoints/          (Phase 4 soup models)
    ├── M1.pth
    ├── M2.pth
    ├── M3.pth
    └── M4.pth

/home/nisalperera/YOLOF/logs/
├── phase_2b.log
├── phase_3.log
├── phase_4.log
├── phase_4b.log
├── phase_6.log
└── phase_7.log
```

---

## Summary

| Phase | Command | Time | Status |
|-------|---------|------|--------|
| 2b | `python yolof_soup/run_phases_2b_to_7.py --phase 2b` | 15–30 min | Automated |
| 3 | `python yolof_soup/run_phases_2b_to_7.py --phase 3` | 45–90 min | Automated |
| 4 | `python yolof_soup/run_phases_2b_to_7.py --phase 4` | 60–120 min | Automated |
| 4b | `python yolof_soup/run_phases_2b_to_7.py --phase 4b` | < 1 min | Automated |
| 5a/5b | (Manual) | 2–4 hr | User-managed |
| 6 | `python yolof_soup/run_phases_2b_to_7.py --phase 6 --after-user-phase-5` | 5–10 min | Automated |
| 7 | `python yolof_soup/run_phases_2b_to_7.py --phase 7 --after-user-phase-5` | 10–20 min | Automated |

**Total automated time:** ~3–5 hours  
**Total user Phase 5 time:** ~2–4 hours  
**Total project time:** ~5–9 hours
