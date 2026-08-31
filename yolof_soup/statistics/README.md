# yolof_soup/statistics -- How to run each test

This directory implements the statistical analyses for the final thesis's
four formal hypotheses (H1-H4, Ch.1 Sec.1.5 / Ch.4 Preliminary Note) plus
two descriptive-only geometric/calibration analyses. See
`STATISTICS_UPDATE_NOTES.md` for the full rationale, including a
cross-chapter numbering inconsistency you should fix in the thesis text.

## Two categories of analysis

| Category | Files | Needs bootstrap manifest? |
|---|---|---|
| Confirmatory (paired image-level bootstrap) | `h1_rq1_static_learning_effect.py`, `h2_rq2_strategy_equivalence.py`, `h3_rq3_m6_vs_m5.py`, `h4_rq4_full_pipeline_superiority.py` | Yes |
| Descriptive-only (no formal hypothesis) | `descriptive_lmc_barrier_geometry.py`, `descriptive_beta_calibration.py` | No |

## One-time setup for the confirmatory tests

All four confirmatory scripts share one bootstrap resampling scheme so
every pairwise contrast in the thesis is paired against the identical
resample. Generate it ONCE, before running any of H1-H4:

```bash
cd /path/to/YOLOF   # repo root

pip install numpy scipy pycocotools

python -m yolof_soup.statistics.generate_coco_bootstrap_manifest \
  --ground-truth results/ground_truth_heldout.json \
  --predictions results/predictions/condition_1.json \
               results/predictions/condition_2.json \
               results/predictions/condition_3.json \
               results/predictions/condition_4.json \
               results/predictions/condition_5.json \
               results/predictions/condition_6.json \
               results/predictions/L1.json \
               results/predictions/D1.json \
               results/predictions/D2.json \
               results/predictions/C3.json \
  --iterations 5000 \
  --seed 42 \
  --output-dir results/bootstrap_manifest
```

Do a smoke test first with `--iterations 100` to confirm every file validates.

## Running each test individually

### H1/RQ1 -- Static learning effect

Compares M6 vs Condition 1 (global uniform) and M6 vs the best individual
ingredient. Config: `configs/h1_predictions.json` with keys `condition_1`,
`condition_6`, `best_ingredient`.

```bash
python -m yolof_soup.statistics.h1_rq1_static_learning_effect \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h1_predictions.json \
  --workers 4
```

Output: `results/h1_rq1_results.json` (+ timestamped copy), `results/h1_rq1_bootstrap/`.

### H2/RQ2 -- Strategy equivalence (Conditions 2-6)

Planned pairwise contrasts: Condition 2 vs {3, 4, 5, 6}, and Condition 3
vs 4. Excludes M6 vs M5 (that's H3). Config: `configs/h2_predictions.json`
with keys `condition_2` through `condition_6`.

```bash
python -m yolof_soup.statistics.h2_rq2_strategy_equivalence \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h2_predictions.json \
  --coefficients-file results/phase3_soup_results.json \
  --workers 4
```

Output: `results/h2_rq2_results.json`, `results/h2_rq2_bootstrap/`.

### H3/RQ3 -- M6 vs M5

Config: `configs/h3_predictions.json` with keys `condition_5`, `condition_6`.

```bash
python -m yolof_soup.statistics.h3_rq3_m6_vs_m5 \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h3_predictions.json \
  --workers 4
```

Output: `results/h3_rq3_results.json`, `results/h3_rq3_bootstrap/`.

### H4/RQ4 -- Full pipeline superiority

Confirmatory: C3 vs best individual ingredient. Descriptive: C3 vs
published baseline (37.7 AP), fine-tuning gain pairs, D2 vs C3. Config:
`configs/h4_predictions.json` with keys `condition_2`, `condition_6`,
`best_condition_3to5`, `best_ingredient`, `D1`, `D2`, `C3`.

```bash
python -m yolof_soup.statistics.h4_rq4_full_pipeline_superiority \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h4_predictions.json \
  --workers 4
```

Output: `results/h4_rq4_results.json`, `results/h4_rq4_bootstrap/`.

### Descriptive -- LMC barrier geometry

No bootstrap manifest needed. Requires `results/phase4_barrier_results.json`
(90-row base-conditioned barrier data, see script docstring for schema).

```bash
python -m yolof_soup.statistics.descriptive_lmc_barrier_geometry
```

Output: `results/descriptive_lmc_barrier_geometry_results.json`,
raw and pair-averaged audit tables.

### Descriptive -- Beta temperature calibration

No bootstrap manifest needed. Requires `results/phase3_soup_results.json`
with `condition_6.beta_values`.

```bash
python -m yolof_soup.statistics.descriptive_beta_calibration
```

Output: `results/descriptive_beta_calibration_results.json`.

## Running everything at once

```bash
python -m yolof_soup.statistics.run_all_stats
```

This runs all six analyses in order, skips any whose input files are
missing (rather than failing the whole run), and writes
`results/statistical_summary.json`.

## Superseded files (not deleted, not run by the orchestrator)

`h1_rq1_component_vs_global.py`, `h3_rm_anova_conditions2to5.py`,
`h4a_paired_ttest_M6_vs_M5.py`, `h4c_finetune_gain.py`,
`h2_rq2_lmc_barrier_anova.py`, `h4b_beta_onesample_ttest.py` are earlier
versions kept for reference. They are not imported by
`run_all_stats.py`. Delete them once you've confirmed the renamed files
produce the results you expect.
