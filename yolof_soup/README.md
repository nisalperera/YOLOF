# YOLOF Soup Module Guide

This guide covers the current thesis-oriented soup workflow implemented in `yolof_soup`. The downstream experiments are implemented as Python modules, while the phase 1 and phase 2 training artifacts are treated as config-backed prerequisites for those experiments.

## 1. What This Module Does

`yolof_soup` orchestrates the model-soup workflow around YOLOF checkpoints and analysis artifacts:

- Phase 1: backbone and encoder training outputs
- Phase 2: decoder sweep outputs and ingredient checkpoints
- Phase 3: soup construction and evaluation
- Phase 4: loss-barrier and sharpness analysis
- Phase 5: cross-domain evaluation on Pascal VOC 2007
- RQ1 and RQ3 statistical summaries

All shared paths, dataset names, checkpoints, and evaluation settings live in [config/experiment_config.py](config/experiment_config.py) and [config/experiment_registry.py](config/experiment_registry.py).

## 2. Module Layout

- [config/experiment_config.py](config/experiment_config.py): central paths, datasets, checkpoints, defaults, and Detectron2 config factories
- [config/experiment_registry.py](config/experiment_registry.py): canonical run IDs and merge-condition metadata
- [experiments/phase3_soup_construction.py](experiments/phase3_soup_construction.py): soup construction and M1-M4 evaluation
- [experiments/phase4_loss_landscape.py](experiments/phase4_loss_landscape.py): decoder/backbone barriers, branch comparisons, and sharpness statistics
- [experiments/phase5_cross_domain.py](experiments/phase5_cross_domain.py): COCO versus VOC transfer comparison
- [experiments/rq1_final_test.py](experiments/rq1_final_test.py): Wilcoxon and effect-size summary for RQ1/H1
- [experiments/rq3_diversity_analysis.py](experiments/rq3_diversity_analysis.py): diversity versus performance analysis for RQ3/H3
- [utils/](utils/): checkpoint, key-partitioning, evaluation, and statistics helpers
- [run_all_experiments.sh](run_all_experiments.sh): sequential runner for the current downstream phases

The legacy scripts under [../tools/](../tools/) were useful during development, but the supported entry points for this workflow are the modules under [experiments/](experiments/).

## 2.1 Legacy Script Migration

If you previously used the legacy scripts in [../tools/](../tools/), use this mapping:

- `python tools/build_soup.py ...` -> `python -m yolof_soup.experiments.phase3_soup_construction`
- `python tools/analyze_connectivity.py ...` -> `python -m yolof_soup.experiments.phase4_loss_landscape`
- `python tools/analyze_loss_landscape.py ...` -> `python -m yolof_soup.experiments.phase4_loss_landscape` for barrier and sharpness analysis
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

From the repository root:

```bash
bash yolof_soup/run_all_experiments.sh
```

This runs, in order:

1. Phase 3 soup construction
2. Phase 4 loss-landscape analysis
3. Phase 5 cross-domain evaluation
4. RQ1 final statistical test
5. RQ3 diversity analysis

`PYTHON_BIN` defaults to `python` and can be overridden for a different interpreter.

## 7. Run Individual Phases

From the repository root:

```bash
python -m yolof_soup.experiments.phase3_soup_construction
python -m yolof_soup.experiments.phase4_loss_landscape
python -m yolof_soup.experiments.phase5_cross_domain
python -m yolof_soup.experiments.rq1_final_test
python -m yolof_soup.experiments.rq3_diversity_analysis
```

Recommended order is the same as above.

Phase 1 and Phase 2 are not exposed as standalone modules in this repository. Their outputs are the checkpoints consumed by the downstream experiment scripts above.

## 8. What Each Phase Produces

### Phase 3: `phase3_soup_construction`

Primary outputs:

- `results/experiment_run_manifest.json`
- `results/phase3_soup_results.json`
- `checkpoints/learned_head_soup.pth`
- `checkpoints/global_uniform_soup.pth`
- `checkpoints/branch_uniform_soup.pth`
- `checkpoints/branch_dirichlet_soup.pth`
- `checkpoints/branch_fisher_soup.pth`

This phase builds the baseline, greedy, learned, and branch-weighted soups and evaluates the M1-M4 conditions.

### Phase 4: `phase4_loss_landscape`

Primary output:

- `results/phase4_landscape_results.json`

This file includes:

- decoder pair barriers
- branch barriers for `cls`, `reg`, `shared`, and `full_decoder`
- branch comparison summaries and paired deltas
- backbone pair barriers and summary statistics
- decoder and backbone sharpness values
- Mann-Whitney, Wilcoxon, and Spearman statistics where available

### Phase 5: `phase5_cross_domain`

Primary output:

- `results/phase5_cross_domain_results.json`

This file includes:

- COCO in-domain metrics for head and global soups
- VOC cross-domain metrics for head and global soups
- `delta_coco` and `delta_voc`
- bootstrap confidence intervals
- the H4 direction summary

### RQ1 Final Test: `rq1_final_test`

Primary output:

- `results/rq1_test_results.json`

This file contains the Wilcoxon result and Cohen's d summary.

### RQ3 Diversity Analysis: `rq3_diversity_analysis`

Primary output:

- `results/rq3_diversity_results.json`

This file contains N=3 versus N=6 diversity and mAP comparisons for head and global soups, plus the H3 result.

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
python -m unittest tests.test_phase4_result_schema
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
3. Run Phase 3 and inspect `results/phase3_soup_results.json`.
4. Run Phase 4 for barriers, branch comparisons, and sharpness.
5. Run Phase 5 for cross-domain transfer checks.
6. Run the RQ1 and RQ3 scripts for compact hypothesis summaries.
7. Archive `results/*.json` and the corresponding checkpoints for reproducibility.