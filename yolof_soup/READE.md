# YOLOF Soup Module Guide

This document explains how to use the `yolof_soup` module end-to-end for thesis-style model soup experiments.

## 1. What This Module Does

`yolof_soup` orchestrates a multi-phase workflow around YOLOF checkpoints:

- Build and evaluate soup variants (Phase 3)
- Analyze loss barriers and sharpness (Phase 4)
- Evaluate cross-domain transfer on VOC (Phase 5)
- Run statistical hypothesis tests for RQ1 and RQ3

It is designed to use a shared config source (`config/experiment_config.py`) and save reproducible artifacts in `checkpoints/` and `results/`.

## 2. Module Layout

- `config/experiment_config.py`: central paths, datasets, thresholds, and cfg factories
- `config/experiment_registry.py`: canonical run IDs and merge condition IDs
- `experiments/phase3_soup_construction.py`: soup construction + condition M1-M4 eval
- `experiments/phase4_loss_landscape.py`: barriers, branch barriers, sharpness, statistics
- `experiments/phase5_cross_domain.py`: COCO vs VOC transfer comparison
- `experiments/rq1_final_test.py`: Wilcoxon + effect size summary for RQ1/H1
- `experiments/rq3_diversity_analysis.py`: diversity vs performance analysis for RQ3/H3
- `utils/`: checkpoint, key partitioning, eval, and statistics utilities
- `run_all_experiments.sh`: sequential runner for all phases

## 3. Prerequisites

## Python and dependencies

Use the repository root environment and install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies used by this module include:

- PyTorch
- Detectron2
- NumPy
- pyhessian (for Fisher/Hessian branch weighting path in M4)

## Dataset expectations

The module expects COCO and VOC paths/settings from `experiment_config.py`:

- COCO root: `COCO_ROOT` (default under `yolof_soup/data/coco`)
- Selection split annotations: `instances_val2017_selection2000.json`
- Eval split annotations: `instances_val2017_eval4000.json`
- VOC annotation file in COCO format

If annotation files are missing, dataset registration is skipped with warnings.

## Checkpoint expectations

Before running phases, these are typically expected:

- Backbone/encoder checkpoint (`BACKBONE_ENC_CKPT`)
- Decoder ingredient checkpoints (`DECODER_CKPT_PATHS`)
- Full-model ingredient checkpoints (`GLOBAL_CKPT_PATHS`)
- Baseline checkpoint (`BASELINE_CKPT`)

All defaults are defined in `config/experiment_config.py`.

## 4. Configuration and Environment Variables

You can override key defaults before running experiments:

```bash
export THESIS_ROOT=/path/to/YOLOF/yolof_soup
export THESIS_DEVICE=cuda
export THESIS_NUM_GPUS=1
export THESIS_N_INGREDIENTS=6
export COCO_ROOT=/path/to/coco
export VOC_ROOT=/path/to/voc2007
```

Additional config behavior:

- directories are auto-created on import (`checkpoints`, `results`, `logs`)
- Detectron2 cfg factories are provided via `build_eval_cfg(...)`
- datasets are registered idempotently when Detectron2 is available

## 5. Run IDs and Merge Conditions

The registry keeps run IDs separate from merge condition IDs:

- Ingredient/full runs: `L1-L4`, `C1-C2`
- Downstream/final runs: `D1-D2`, `C3`
- Merge conditions: `M1-M4`

Merge condition meanings:

- `M1`: global uniform soup over full model
- `M2`: branch-uniform soup (cls/reg/shared partitioning)
- `M3`: branch Dirichlet search over cls/reg coefficients
- `M4`: Fisher/Hessian branch-weighted soup (pyhessian-first, fallback proxy)

## 6. Quick Start

From repository root:

```bash
bash yolof_soup/run_all_experiments.sh
```

This runs:

1. Phase 3 soup construction
2. Phase 4 loss landscape
3. Phase 5 cross-domain analysis
4. RQ1 final statistical test
5. RQ3 diversity analysis

## 7. Run Individual Phases

From repository root:

```bash
python -m yolof_soup.experiments.phase3_soup_construction
python -m yolof_soup.experiments.phase4_loss_landscape
python -m yolof_soup.experiments.phase5_cross_domain
python -m yolof_soup.experiments.rq1_final_test
python -m yolof_soup.experiments.rq3_diversity_analysis
```

Recommended order is the same as above.

## 8. What Each Phase Produces

## Phase 3 (`phase3_soup_construction`)

Primary outputs:

- `results/phase3_soup_results.json`
- `results/experiment_run_manifest.json`
- checkpoints for learned and condition soups, including:
  - `learned_head_soup.pth`
  - `global_uniform_soup.pth`
  - `branch_uniform_soup.pth`
  - `branch_dirichlet_soup.pth`
  - `branch_fisher_soup.pth`

Includes baseline, uniform/greedy/learned head soup metrics, and M1-M4 results.

## Phase 4 (`phase4_loss_landscape`)

Primary output:

- `results/phase4_landscape_results.json`

Contains:

- decoder pair barriers
- branch barriers (`cls`, `reg`, `shared`, `full_decoder`)
- branch summaries and branch comparison deltas
- backbone pair barriers and summary
- sharpness values
- statistical outputs (Mann-Whitney, Wilcoxon, Spearman where available)

## Phase 5 (`phase5_cross_domain`)

Primary output:

- `results/phase5_cross_domain_results.json`

Contains in-domain COCO and cross-domain VOC comparisons, deltas, CIs, and H4 direction.

## RQ1 Final Test (`rq1_final_test`)

Primary output:

- `results/rq1_test_results.json`

Contains Wilcoxon result and Cohen's d summary.

## RQ3 Diversity Analysis (`rq3_diversity_analysis`)

Primary output:

- `results/rq3_diversity_results.json`

Contains N=3 vs N=6 diversity and mAP comparisons for head/global soups, plus H3 result.

## 9. Programmatic Usage Examples

## Build eval config

```python
from yolof_soup.config.experiment_config import build_eval_cfg, EVAL_DATASET

cfg = build_eval_cfg(EVAL_DATASET)
```

## Access registry metadata

```python
from yolof_soup.config.experiment_registry import ingredient_run_ids, get_merge_conditions

print(ingredient_run_ids())
print([c.condition_id for c in get_merge_conditions()])
```

## 10. Validation and Testing

Run selected tests from repository root:

```bash
python -m unittest tests.test_experiment_registry
python -m unittest tests.test_model_soup_strategy_smoke
python -m unittest tests.test_fisher_coefficients
python -m unittest tests.test_phase4_result_schema
```

Notes:

- some tests skip automatically when torch is unavailable
- full experiment scripts require Detectron2 + datasets + checkpoints

## 11. Troubleshooting

## Import errors for torch or detectron2

Symptom:

- static diagnostics show unresolved imports
- runtime fails on `build_model` or dataset registration

Fix:

- install torch and detectron2 in the active environment
- confirm the interpreter used by shell matches your configured environment

## Dataset registration warnings

Symptom:

- warnings about missing annotation files

Fix:

- verify COCO/VOC annotation JSON paths
- check `COCO_ROOT`, `VOC_ROOT`, and split annotation filenames

## M4 Fisher path fallback

Symptom:

- warning that pyhessian trace failed and proxy fallback was used

Fix:

- install `pyhessian`
- verify model/loss wrapper compatibility and available CUDA resources

## Old unittest traceback referencing `_SPEC.loader.exec_module(...)`

If you still see this traceback after code updates, clear stale cached bytecode and rerun:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
python -m unittest tests.test_experiment_registry
```

## 12. Practical Workflow Recommendation

1. Validate environment and dataset paths.
2. Run Phase 3 and inspect `phase3_soup_results.json`.
3. Run Phase 4 for barriers/sharpness and branch comparisons.
4. Run Phase 5 for cross-domain transfer checks.
5. Run RQ1 and RQ3 scripts for compact hypothesis summaries.
6. Archive `results/*.json` and corresponding checkpoints for reproducibility.

## 13. Notes

- This file is intentionally named `READE.md` to match your requested filename.
- If you prefer the conventional name, add a copy as `README.md` later.
