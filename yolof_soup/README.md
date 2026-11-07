# YOLOF Soup Module Guide

This guide covers the current thesis-oriented soup workflow implemented in `yolof_soup`. The downstream experiments are implemented as Python modules, while the phase 1 and phase 2 training artifacts are treated as config-backed prerequisites for those experiments.

## 1. What This Module Does

`yolof_soup` orchestrates the model-soup workflow around YOLOF checkpoints and analysis artifacts:

- Phase 1: backbone and encoder training outputs
- Phase 2: decoder sweep outputs and ingredient checkpoints
- Phase 3: soup construction and evaluation (M1-M4 conditions)
- Phase 4: loss-landscape analysis (LMC barriers, Hessian traces, statistical tests)
- Phase 5: head fine-tuning evaluation (D1, D2, C3 variants)
- RQ3: coefficient strategy comparison tests
- RQ4: full pipeline and per-category performance analysis

All shared paths, dataset names, checkpoints, and evaluation settings live in [config/experiment_config.py](config/experiment_config.py) and [config/experiment_registry.py](config/experiment_registry.py).

## 2. Module Layout

- [config/experiment_config.py](config/experiment_config.py): central paths, datasets, checkpoints, defaults, and Detectron2 config factories
- [config/experiment_registry.py](config/experiment_registry.py): canonical run IDs and merge-condition metadata
- [experiments/soup_construction.py](experiments/soup_construction.py): soup construction and M1-M4 evaluation
- [experiments/loss_landscape.py](experiments/loss_landscape.py): LMC barriers, Hessian traces, and statistical tests
- [experiments/head_finetuning.py](experiments/head_finetuning.py): head-only fine-tuning for D1, D2, C3 variants
- [experiments/coefficient_strategy_test.py](experiments/coefficient_strategy_test.py): coefficient strategy comparison and moderation analysis
- [experiments/full_pipeline_test.py](experiments/full_pipeline_test.py): full pipeline evaluation and per-category analysis
- [experiments/integration_test.py](experiments/integration_test.py): full pipeline integration test with validation
- [utils/](utils/): checkpoint, key-partitioning, evaluation, and statistics helpers
- [run_all_experiments.sh](run_all_experiments.sh): sequential runner for all phases

The legacy scripts under [../tools/](../tools/) were useful during development, but the supported entry points for this workflow are the modules under [experiments/](experiments/).

## 2.1 Legacy Script Migration

If you previously used the legacy scripts in [../tools/](../tools/), use this mapping:

- `python tools/build_soup.py ...` -> `python -m yolof_soup.experiments.soup_construction`
- `python tools/analyze_connectivity.py ...` -> `python -m yolof_soup.experiments.loss_landscape`
- `python tools/analyze_loss_landscape.py ...` -> `python -m yolof_soup.experiments.loss_landscape` for barrier and sharpness analysis
- `python tools/run_all_analysis.py ...` -> `bash yolof_soup/run_all_experiments.sh`

Migration notes:

- The current phase 4 script focuses on interpolation barriers, branch-wise comparisons, and sharpness statistics rather than the older 2-D landscape plot export workflow.
- The current pipeline is registry and manifest driven via [config/experiment_registry.py](config/experiment_registry.py) and the generated `results/experiment_run_manifest.json` artifact, instead of manually passing long checkpoint lists to each tool script.

## 3. Prerequisites

### Python and dependencies

Install the repository dependencies from the project root:

```bash
pip install -r requirements.txt
```

Core dependencies used by this module include:

- PyTorch
- Detectron2
- NumPy
- pyhessian for the Fisher/Hessian branch-weighting path in M4

If Detectron2 is unavailable, dataset registration is skipped with warnings and the experiment modules cannot build models.

### Expected directory layout

By default, the module writes into `yolof_soup/`-relative directories:

- `yolof_soup/checkpoints/`
- `yolof_soup/results/`
- `yolof_soup/logs/`
- `yolof_soup/data/coco/`
- `yolof_soup/data/voc2007/`

You can override the root locations with environment variables before importing the module.

### Dataset expectations

The default dataset configuration expects:

- COCO 2017 images and annotations under `COCO_ROOT` or `yolof_soup/data/coco`
- a training set at `train2017` and `annotations/instances_train2017.json`
- the thesis selection split at `annotations/instances_val2017_selection2000.json`
- the thesis eval split at `annotations/instances_val2017_eval4000.json`
- Pascal VOC 2007 images under `VOC_ROOT` or `yolof_soup/data/voc2007`
- VOC annotations already converted to COCO format at `annotations/voc2007_test_coco_format.json`

The selection and eval splits are registered idempotently when their annotation files exist. If those files are missing, registration is skipped and the module logs a warning.

### Checkpoint expectations

The downstream phases consume the following checkpoints and output directories from [config/experiment_config.py](config/experiment_config.py):

- pretrained COCO weights: `checkpoints/pretrained/yolof_R_50_C5_1x.pth`
- Phase 1 output: `checkpoints/phase1_backbone_encoder/backbone_encoder.pth`
- Phase 2 decoder ingredients: `checkpoints/phase2_decoder_sweep/decoder_run00.pth` through `decoder_run05.pth` by default
- Phase 2 full-model ingredients: `checkpoints/phase2_decoder_sweep/global_run00.pth` through `global_run05.pth` by default
- best baseline checkpoint: `checkpoints/phase2_decoder_sweep/best_individual.pth`

The default number of ingredient runs is `N_INGREDIENTS = 6`. If you change it, the registry and downstream phase inputs must stay aligned.

## 4. Configuration and Environment Variables

Set these before running the experiments if you need to override the defaults:

```bash
export THESIS_ROOT=/path/to/YOLOF/yolof_soup
export THESIS_DEVICE=cuda
export THESIS_NUM_GPUS=1
export THESIS_N_INGREDIENTS=6
export COCO_ROOT=/path/to/coco
export VOC_ROOT=/path/to/voc2007
```

Additional behavior:

- directories are created on import for checkpoints, results, logs, and the Phase 1/2 output folders
- `build_eval_cfg(...)` returns a frozen Detectron2 config for evaluation-only passes
- dataset registration is idempotent when Detectron2 is available
- `YOLOF_CONFIG_DIR` can be set if your base YOLOF YAML lives outside the repo default path

## 5. Run IDs and Merge Conditions

The registry keeps run IDs separate from merge-condition IDs:

- Ingredient runs: `L1-L4`, `C1-C2`
- Downstream runs: `D1-D2`, `C3`
- Merge conditions: `M1-M4`

Run metadata is defined in [config/experiment_registry.py](config/experiment_registry.py):

- `L1-L4` and `C1-C2` are the six ingredient runs consumed by the soup phases
- `D1-D2` and `C3` are downstream runs used for later pipeline stages
- `M1` is global uniform soup over the full model
- `M2` is branch-uniform soup with separate cls/reg groups
- `M3` is independent Dirichlet search for cls/reg branch weights
- `M4` is Fisher-weighted branch soup using a Hessian-based trace path with a proxy fallback

## 6. Quick Start

### Full Integration Test

From the repository root, run the complete integration test with validation:

```bash
bash yolof_soup/run_all_experiments.sh --integration
```

This performs:

1. Input validation (checkpoints, datasets, directories)
2. Phase 3: soup construction (M1-M4)
3. Phase 4: loss-landscape analysis
4. Phase 5: head fine-tuning (D1, D2, C3)
5. RQ3: coefficient strategy tests
6. RQ4: full pipeline and per-category analysis
7. Integration report generation

### Standard Mode

From the repository root:

```bash
bash yolof_soup/run_all_experiments.sh
```

This runs all phases sequentially (without integration validation).

## 7. Run Individual Phases

From the repository root:

```bash
# Individual phases
python -m yolof_soup.experiments.soup_construction
python -m yolof_soup.experiments.loss_landscape
python -m yolof_soup.experiments.head_finetuning
python -m yolof_soup.experiments.coefficient_strategy_test
python -m yolof_soup.experiments.full_pipeline_test

# Full integration test
python -m yolof_soup.experiments.integration_test

# Integration test with validation only
python -m yolof_soup.experiments.integration_test --validate-only

# Resume from specific phase
python -m yolof_soup.experiments.integration_test --resume-from phase5
```

Recommended order is Phase 3 → Phase 4 → Phase 5 → RQ3 → RQ4 for proper data dependencies.

Phase 1 and Phase 2 are not exposed as standalone modules in this repository. Their outputs are the checkpoints consumed by the downstream experiment scripts above.

## 8. What Each Phase Produces

### Phase 3: `soup_construction`

Primary outputs:

- `results/experiment_run_manifest.json` - Registry of all ingredient runs
- `results/soup_results.json` - Per-class AP for M1-M4 conditions
- `checkpoints/m1_global_uniform_soup.pth` - Global uniform average
- `checkpoints/m2_branch_uniform_soup.pth` - Branch-uniform averaging
- `checkpoints/m3_dirichlet_soup.pth` - Coordinate-descent learned soup
- `checkpoints/m4_fisher_weighted_soup.pth` - Fisher-weighted soup

This phase builds 4 merging conditions (M1-M4) and evaluates on held-out split.

### Phase 4: `loss_landscape`

Primary outputs:

- `results/lmc_barriers.json` - LMC barrier heights for all ingredient pairs
- `results/hessian_traces.json` - Hessian trace estimates (L2-norm proxy)
- `results/statistical_tests.json` - RM-ANOVA statistical test results

This phase measures loss landscape geometry (barriers, sharpness) and runs statistical tests on component-wise analysis.

### Phase 5: `head_finetuning`

Primary outputs:

- `checkpoints/d1_finetuned.pth` - Fine-tuned from M2 (global uniform)
- `checkpoints/d2_finetuned.pth` - Fine-tuned from best learned soup
- `checkpoints/c3_pipeline.pth` - Full pipeline variant
- `results/finetuning_results.json` - Per-class AP and training metrics

This phase fine-tunes decoder heads on 3 variants and compares in-domain performance.

### RQ3: `coefficient_strategy_test`

Primary outputs:

- `results/test_results.json` - Contains:
  - Test 1: Paired t-test and Wilcoxon (M3 vs M4 strategy comparison)
  - Test 2: D1 vs D2 fine-tuning moderation with 95% CI
  - Test 3: Two-way RM-ANOVA framework (ready for learned coefficient data)

Tests strategy selection (Dirichlet vs Fisher) and fine-tuning effects.

### RQ4: `full_pipeline_test`

Primary outputs:

- `results/test_results.json` - Contains:
  - Test 1: Bootstrap CI for full pipeline (C3)
  - Test 2: One-sample t-test vs 37.7 AP baseline
  - Test 3: Per-category analysis (rare, small, medium, large objects)
  - Test 4: One-way ANOVA (D1 vs D2 vs C3) with post-hoc tests

Evaluates overall pipeline efficacy and per-category performance.

### Integration Test: `integration_test`

Primary outputs:

- `results/integration_test_report.txt` - Comprehensive integration report with:
  - Phase status (pass/fail)
  - Generated artifact validation
  - Recommendations for next steps

Validates entire pipeline and generates diagnostic report.

## 9. Programmatic Usage Examples

### Build an evaluation config

```python
from yolof_soup.config.experiment_config import build_eval_cfg, EVAL_DATASET

cfg = build_eval_cfg(EVAL_DATASET)
```

### Inspect registry metadata

```python
from yolof_soup.config.experiment_registry import ingredient_run_ids, get_merge_conditions

print(ingredient_run_ids())
print([c.condition_id for c in get_merge_conditions()])
```

## 10. Validation and Testing

Run the focused tests from the repository root:

```bash
python -m unittest tests.test_experiment_registry
python -m unittest tests.test_model_soup_strategy_smoke
python -m unittest tests.test_fisher_coefficients
python -m unittest tests.test_result_schema
```

Notes:

- some tests skip automatically when torch is unavailable
- the full experiment scripts require Detectron2, datasets, and checkpoints

## 11. Troubleshooting

### Import errors for torch or detectron2

Symptom:

- unresolved imports in static analysis
- runtime failures when building the model or registering datasets

Fix:

- install torch and Detectron2 in the active environment
- confirm the shell interpreter matches the environment used by VS Code

### Dataset registration warnings

Symptom:

- warnings about missing annotation files

Fix:

- verify the COCO and VOC annotation JSON paths
- confirm `COCO_ROOT`, `VOC_ROOT`, and the thesis split filenames

### M4 Fisher fallback

Symptom:

- a warning that the pyhessian trace failed and a proxy fallback was used

Fix:

- install `pyhessian`
- verify model and loss-wrapper compatibility
- check that enough CUDA resources are available for the trace path

### Stale unittest bytecode

If you still see an old traceback after code updates, clear cached bytecode and rerun:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
python -m unittest tests.test_experiment_registry
```

## 12. Practical Workflow Recommendation

1. Validate environment variables and dataset paths.
2. Confirm the Phase 1 and Phase 2 prerequisite checkpoints exist.
3. Run Phase 3 and inspect `results/soup_results.json`.
4. Run Phase 4 for barriers, branch comparisons, and sharpness.
5. Run Phase 5 for cross-domain transfer checks.
6. Run the RQ1 and RQ3 scripts for compact hypothesis summaries.
7. Archive `results/*.json` and the corresponding checkpoints for reproducibility.