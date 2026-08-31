# yolof_soup/statistics -- How to run each test

This directory implements the statistical analyses for the final thesis's
four formal hypotheses (H1-H4, Chapter 1 Section 1.5 and Chapter 4
Preliminary Note) plus two descriptive-only geometric/calibration analyses.

The confirmatory analyses use a paired image-level bootstrap. The resampling
unit is one held-out COCO evaluation image, with all of its annotations and
saved detections. Do not treat COCO categories, per-class AP values, or
individual bounding boxes as independent observations. See
`STATISTICS_UPDATE_NOTES.md` for the final numbering mapping and rationale.

## Analysis inventory

| Category | Script | Purpose | Needs bootstrap manifest? |
|---|---|---|---|
| Confirmatory | `h1_rq1_static_learning_effect.py` | H1/RQ1: M6 vs Condition 1; M6 vs best ingredient | Yes |
| Confirmatory | `h2_rq2_strategy_equivalence.py` | H2/RQ2: strategy contrasts across Conditions 2-6 | Yes |
| Confirmatory | `h3_rq3_m6_vs_m5.py` | H3/RQ3: M6 vs M5 | Yes |
| Confirmatory | `h4_rq4_full_pipeline_superiority.py` | H4/RQ4: C3 vs best ingredient | Yes |
| Descriptive only | `descriptive_lmc_barrier_geometry.py` | Per-component LMC barrier geometry | No |
| Descriptive only | `descriptive_beta_calibration.py` | M6 temperature scalars | No |

## Shared input requirements

All paths below are relative to the repository root. Run commands from the
repository root, not from `yolof_soup/statistics/`:

```bash
cd /path/to/YOLOF
```

Install the common dependencies:

```bash
pip install numpy scipy pycocotools
```

### 1. Held-out COCO ground truth

**Required by:** H1, H2, H3, H4, and bootstrap-manifest generation.

**Default path:**

```text
results/ground_truth_heldout.json
```

This must be a valid COCO ground-truth annotation JSON for the final held-out
COCO val2017 partition. It should contain approximately 4,047 images in this
thesis design. Keep original COCO `image_id`, `category_id`, and annotation
IDs. Do not remap categories and do not include calibration images.

Required top-level fields:

```json
{
  "images": [
    {
      "id": 397133,
      "file_name": "000000397133.jpg",
      "width": 640,
      "height": 427
    }
  ],
  "annotations": [
    {
      "id": 123456,
      "image_id": 397133,
      "category_id": 1,
      "bbox": [10.5, 20.0, 120.0, 250.0],
      "area": 30000.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

Validation requirements:

- Every `images[*].id` must be unique.
- Every `annotations[*].image_id` must occur in `images[*].id`.
- Preserve the official COCO category ID space. These IDs are non-contiguous,
  for example `person=1`, `car=3`, `stop sign=13`, `toothbrush=90`.
- Use the same exact ground-truth file for every confirmatory comparison.
- Do not use this set for coefficient selection, temperature calibration,
  checkpoint selection, or threshold tuning.

### 2. COCO detection prediction results

**Required by:** H1, H2, H3, H4, and bootstrap-manifest generation.

Each model must have one already-generated COCO detection-result JSON. The
statistics scripts do not run model inference. They reuse these predictions
and recalculate COCO AP50:95 on bootstrap-resampled image sets.

Recommended location:

```text
results/predictions/
```

Example filenames:

```text
results/predictions/condition_1.json
results/predictions/condition_2.json
results/predictions/condition_3.json
results/predictions/condition_4.json
results/predictions/condition_5.json
results/predictions/condition_6.json
results/predictions/L1.json
results/predictions/D1.json
results/predictions/D2.json
results/predictions/C3.json
```

A prediction file must be a JSON **list**, not a dictionary. Every detection
must contain `image_id`, `category_id`, `bbox`, and `score`:

```json
[
  {
    "image_id": 397133,
    "category_id": 1,
    "bbox": [12.4, 18.9, 119.7, 249.6],
    "score": 0.9821
  },
  {
    "image_id": 397133,
    "category_id": 3,
    "bbox": [301.3, 170.5, 210.2, 104.1],
    "score": 0.7438
  }
]
```

Field requirements:

| Field | Type | Meaning |
|---|---|---|
| `image_id` | Integer | Must exist in `ground_truth_heldout.json` |
| `category_id` | Integer | Official COCO category ID, not a contiguous model index |
| `bbox` | Four numbers | COCO XYWH box: `[x_min, y_min, width, height]` |
| `score` | Number | Detection confidence, normally in `[0, 1]` |

Important validation checks:

- Every prediction `image_id` must belong to the held-out ground-truth set.
- Empty predictions for an image are valid. Do not fabricate a record for an
  image with no detections.
- All compared models must use the same held-out images, COCO category ID
  space, score threshold, NMS configuration, and evaluator settings.
- Confirm that a direct full-set `COCOeval` result reproduces your stored
  headline AP50:95 before running bootstrap analysis.

### 3. Shared bootstrap manifest

**Required by:** H1, H2, H3, H4.

**Generate once** using `generate_coco_bootstrap_manifest.py`, then reuse
exactly the same output file for every confirmatory test. This makes each
model comparison paired on identical bootstrap image draws.

Smoke test first:

```bash
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
  --prediction-names Condition1 Condition2 Condition3 Condition4 M5 M6 L1 D1 D2 C3 \
  --iterations 100 \
  --seed 42 \
  --output-dir results/bootstrap_smoke_test
```

Final manifest:

```bash
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
  --prediction-names Condition1 Condition2 Condition3 Condition4 M5 M6 L1 D1 D2 C3 \
  --iterations 5000 \
  --seed 42 \
  --output-dir results/bootstrap_manifest
```

Expected files in `results/bootstrap_manifest/`:

| File | Format | Purpose |
|---|---|---|
| `bootstrap_image_ids.npy` | NumPy `int64` array, shape `(B, N)` | The required input for H1-H4. Each row is one image-ID resample with replacement. |
| `bootstrap_draws.npy` | NumPy `int32` array, shape `(B, N)` | Row entries are indices into the sorted original image-ID vector. |
| `original_heldout_image_ids.txt` | One integer per line | Sorted unique image IDs in ground truth. |
| `bootstrap_summary.csv` | CSV | Per-replicate unique/duplicate image counts. |
| `bootstrap_manifest.json` | JSON object | Seed, checksums, input paths, image count, and prediction-coverage audit. |

Definitions:

- `B` is the number of bootstrap replicates: use `B=5000` for final results.
- `N` is the held-out image count: `N=4047` in the thesis protocol.
- Each bootstrap row has length `N` and samples with replacement. Duplicate
  image IDs are expected and required. A simple shuffled image list is not a
  bootstrap resample.
- Each resample normally has about 63.2% distinct original images. The rest
  are duplicate occurrences, while other original images are omitted.

Do not edit, regenerate, or mix bootstrap manifests after inspecting model
results. Keep the chosen manifest seed fixed at 42 and record its SHA-256
checksum in the experimental log.

## Prediction-config file format

The H1-H4 scripts each take a small JSON configuration file mapping a fixed
model label to an existing COCO prediction JSON file. Create the `configs/`
directory if it does not exist:

```bash
mkdir -p configs
```

Paths may be relative to the repository root or absolute. Use forward slashes
on Linux. Every path must point to the JSON list format described above.

Do not select a "best" model using the held-out set. If `best_ingredient` or
`best_condition_3to5` was selected by maximum held-out AP, label the related
comparison as exploratory post-selection inference in the thesis. The
preferred procedure is selection on the separate 953-image calibration split.

## Running each test individually

### H1/RQ1: Static learning effect

**Question:** Does M6 outperform Condition 1 (global uniform soup) and the
best individual ingredient model?

**Script:** `h1_rq1_static_learning_effect.py`

**Required input files:**

| Input | Default path | Format |
|---|---|---|
| Held-out ground truth | `results/ground_truth_heldout.json` | COCO ground-truth JSON |
| Bootstrap draws | `results/bootstrap_manifest/bootstrap_image_ids.npy` | NumPy array `(5000, 4047)` for final analysis |
| H1 model-path config | `configs/h1_predictions.json` | JSON object below |

**`configs/h1_predictions.json` format:**

```json
{
  "condition_1": "results/predictions/condition_1.json",
  "condition_6": "results/predictions/condition_6.json",
  "best_ingredient": "results/predictions/L1.json"
}
```

Key meanings:

| Key | Required | Prediction source |
|---|---|---|
| `condition_1` | Yes | Global uniform soup, C1 |
| `condition_6` | Yes | M6, tri-component learned soup |
| `best_ingredient` | Yes | Ingredient selected on calibration split, for example L1 or L4 |

Run:

```bash
python -m yolof_soup.statistics.h1_rq1_static_learning_effect \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h1_predictions.json \
  --workers 4
```

Outputs:

```text
results/h1_rq1_results.json
results/h1_rq1_results_YYYY-MM-DD_HH-MM-SS.json
results/h1_rq1_bootstrap/comparison_1_bootstrap_differences.npy
results/h1_rq1_bootstrap/comparison_2_bootstrap_differences.npy
```

The test uses one-sided paired bootstrap p-values for M6 superiority and
Holm-Bonferroni adjustment across its two H1 comparisons. The H1 practical
threshold is `ΔAP50:95 >= 0.5` AP points.

### H2/RQ2: Strategy equivalence

**Question:** Are any coefficient-learning strategy differences across
Conditions 2-6 both statistically significant and practically meaningful
at or above 0.5 AP points?

**Script:** `h2_rq2_strategy_equivalence.py`

**Required input files:**

| Input | Default path | Format |
|---|---|---|
| Held-out ground truth | `results/ground_truth_heldout.json` | COCO ground-truth JSON |
| Bootstrap draws | `results/bootstrap_manifest/bootstrap_image_ids.npy` | NumPy array `(B, N)` |
| H2 model-path config | `configs/h2_predictions.json` | JSON object below |
| Coefficient result file | `results/phase3_soup_results.json` | Optional for Test 3 only, schema below |

**`configs/h2_predictions.json` format:**

```json
{
  "condition_2": "results/predictions/condition_2.json",
  "condition_3": "results/predictions/condition_3.json",
  "condition_4": "results/predictions/condition_4.json",
  "condition_5": "results/predictions/condition_5.json",
  "condition_6": "results/predictions/condition_6.json"
}
```

All five keys are mandatory. Their prediction files must be in standard COCO
result-list format.

Planned bootstrap contrasts:

```text
Condition 2 -> Condition 3
Condition 2 -> Condition 4
Condition 2 -> Condition 5 (M5)
Condition 2 -> Condition 6 (M6)
Condition 3 -> Condition 4
```

M6 vs M5 is excluded here because it is tested separately under H3/RQ3.
Holm-Bonferroni adjusts the five one-sided p-values in this H2 family.

Run:

```bash
python -m yolof_soup.statistics.h2_rq2_strategy_equivalence \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h2_predictions.json \
  --coefficients-file results/phase3_soup_results.json \
  --workers 4
```

Outputs:

```text
results/h2_rq2_results.json
results/h2_rq2_bootstrap/contrast_1_bootstrap_differences.npy
...
results/h2_rq2_bootstrap/contrast_5_bootstrap_differences.npy
```

#### Optional H2 Test 3 coefficient input format

`phase3_soup_results.json` is only required for the supplementary coefficient
magnitude analysis. If it does not exist or lacks the required fields, the
script skips Test 3 and still runs Test 1.

Minimum required schema:

```json
{
  "condition_3": {
    "coefficients": {
      "cls":  [0.15, 0.12, 0.18, 0.20, 0.17, 0.18],
      "bbox": [0.19, 0.16, 0.15, 0.21, 0.14, 0.15],
      "obj":  [0.16, 0.17, 0.16, 0.18, 0.17, 0.16]
    }
  },
  "condition_4": {
    "coefficients": {
      "cls":  [0.14, 0.13, 0.19, 0.19, 0.16, 0.19],
      "bbox": [0.20, 0.15, 0.16, 0.20, 0.14, 0.15],
      "obj":  [0.15, 0.18, 0.16, 0.17, 0.18, 0.16]
    }
  }
}
```

Requirements:

- `condition_3.coefficients` and `condition_4.coefficients` must exist.
- Each must have `cls`, `bbox`, and `obj` lists.
- Each list must have exactly `N=6` numeric values, aligned to ingredient
  ordering L1, L2, L3, L4, R1, R2.
- Test 3's unit is the six ingredient models, not COCO categories.

### H3/RQ3: M6 vs M5

**Question:** Does the tri-component learned soup M6 outperform M5, which
uses one shared coefficient and temperature pair?

**Script:** `h3_rq3_m6_vs_m5.py`

**Required input files:**

| Input | Default path | Format |
|---|---|---|
| Held-out ground truth | `results/ground_truth_heldout.json` | COCO ground-truth JSON |
| Bootstrap draws | `results/bootstrap_manifest/bootstrap_image_ids.npy` | NumPy array `(B, N)` |
| H3 model-path config | `configs/h3_predictions.json` | JSON object below |

**`configs/h3_predictions.json` format:**

```json
{
  "condition_5": "results/predictions/condition_5.json",
  "condition_6": "results/predictions/condition_6.json"
}
```

| Key | Required | Prediction source |
|---|---|---|
| `condition_5` | Yes | M5, shared learned α + β |
| `condition_6` | Yes | M6, independent component-specific α + β |

Run:

```bash
python -m yolof_soup.statistics.h3_rq3_m6_vs_m5 \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h3_predictions.json \
  --workers 4
```

Outputs:

```text
results/h3_rq3_results.json
results/h3_rq3_bootstrap/m6_vs_m5_bootstrap_differences.npy
```

This is one planned directional contrast. The output reports M6 minus M5
AP50:95, a percentile 95% confidence interval, and a one-sided bootstrap
p-value.

### H4/RQ4: Full pipeline superiority

**Question:** Does the full pipeline, M6 followed by C3 decoder-only
fine-tuning, outperform the best individual ingredient model by at least
0.5 AP points?

**Script:** `h4_rq4_full_pipeline_superiority.py`

**Required input files:**

| Input | Default path | Format |
|---|---|---|
| Held-out ground truth | `results/ground_truth_heldout.json` | COCO ground-truth JSON |
| Bootstrap draws | `results/bootstrap_manifest/bootstrap_image_ids.npy` | NumPy array `(B, N)` |
| H4 model-path config | `configs/h4_predictions.json` | JSON object below |

**`configs/h4_predictions.json` format:**

```json
{
  "condition_2": "results/predictions/condition_2.json",
  "condition_6": "results/predictions/condition_6.json",
  "best_condition_3to5": "results/predictions/condition_3.json",
  "best_ingredient": "results/predictions/L1.json",
  "D1": "results/predictions/D1.json",
  "D2": "results/predictions/D2.json",
  "C3": "results/predictions/C3.json"
}
```

Key meanings:

| Key | Used for | Selection requirement |
|---|---|---|
| `condition_2` | Supporting descriptive C2 -> D1 gain | Fixed C2 prediction |
| `condition_6` | Supporting descriptive M6 -> C3 gain | Fixed M6 prediction |
| `best_condition_3to5` | Supporting descriptive best(C3-C5) -> D2 gain | Select on calibration split if possible |
| `best_ingredient` | **Formal H4 comparison:** best ingredient -> C3 | Select on calibration split if possible |
| `D1` | Supporting fine-tuning gain | D1 prediction |
| `D2` | Supporting fine-tuning gain and D2 vs C3 | D2 prediction |
| `C3` | **Formal H4 comparison:** C3 | C3 full pipeline prediction |

Run:

```bash
python -m yolof_soup.statistics.h4_rq4_full_pipeline_superiority \
  --ground-truth results/ground_truth_heldout.json \
  --bootstrap-image-ids results/bootstrap_manifest/bootstrap_image_ids.npy \
  --predictions-config configs/h4_predictions.json \
  --workers 4
```

Outputs:

```text
results/h4_rq4_results.json
results/h4_rq4_bootstrap/confirmatory_c3_vs_best_ingredient_bootstrap_differences.npy
results/h4_rq4_bootstrap/supporting_finetune_pair_1_bootstrap_differences.npy
results/h4_rq4_bootstrap/supporting_finetune_pair_2_bootstrap_differences.npy
results/h4_rq4_bootstrap/supporting_finetune_pair_3_bootstrap_differences.npy
results/h4_rq4_bootstrap/supporting_d2_vs_c3_bootstrap_differences.npy
```

The formal H4 decision is based on C3 vs `best_ingredient`: one-sided p
below 0.05, lower confidence bound above zero, and observed gain at least
0.5 AP points. C3 vs the published 37.7 AP YOLOF-R50 benchmark remains a
descriptive contextual comparison because that published scalar is not a
matched prediction file on this held-out subset.

### Descriptive analysis: LMC barrier geometry

**Purpose:** Summarise per-component linear mode connectivity barriers across
all 15 ingredient-model pairs, averaged across the six rotating base-model
conditions. This is descriptive-only in the final thesis and does not make a
formal H2 hypothesis decision.

**Script:** `descriptive_lmc_barrier_geometry.py`

**Required input file:**

```text
results/phase4_barrier_results.json
```

Expected top-level structure:

```json
{
  "0": {
    "pair_0102": {
      "backbone_encoder": 0.0041,
      "cls_head": 0.0052,
      "reg_head": 0.0084,
      "shared": 0.0020,
      "full_model": 0.0067
    },
    "pair_0103": {
      "backbone_encoder": 0.0043,
      "cls_head": 0.0051,
      "reg_head": 0.0081,
      "shared": 0.0019,
      "full_model": 0.0065
    }
  },
  "1": {
    "pair_0102": {
      "backbone_encoder": 0.0040,
      "cls_head": 0.0053,
      "reg_head": 0.0082,
      "objectness_module": 0.0021,
      "full_model": 0.0066
    }
  }
}
```

Requirements:

- Exactly six top-level base-model groups: `"0"` through `"5"`.
- Each base-model group must contain exactly 15 pair records, corresponding
  to the six ingredient models' `C(6, 2) = 15` unique pairs.
- Every pair name must occur once under every base group.
- Each pair record must contain numeric `backbone_encoder`, `cls_head`,
  `reg_head`, and `full_model` values.
- Objectness may be supplied as either `shared` or `objectness_module`. The
  script normalises `shared` to `objectness_module`.
- Barrier values must all be numeric and expressed on the same scale.

Run:

```bash
python -m yolof_soup.statistics.descriptive_lmc_barrier_geometry
```

Outputs:

```text
results/descriptive_lmc_barrier_raw_90rows.json
results/descriptive_lmc_barrier_pair_averaged_15rows.json
results/descriptive_lmc_barrier_geometry_results.json
```

The script produces descriptive RM-ANOVA/post-hoc and cls-vs-reg summaries,
but no reject/fail-to-reject hypothesis decision. Remember that the 15 model
pairs overlap: each ingredient model participates in five pairs, so they are
not fully independent.

### Descriptive analysis: beta temperature calibration

**Purpose:** Report the learned M6 temperature scalars `beta_cls`,
`beta_bbox`, and `beta_obj`. This is descriptive-only in the final thesis;
it does not run a formal one-sample t-test on a single optimisation result.

**Script:** `descriptive_beta_calibration.py`

**Required input file:**

```text
results/phase3_soup_results.json
```

Minimum schema for the normal single-run case:

```json
{
  "condition_6": {
    "beta_values": {
      "cls": 1.0412,
      "bbox": 0.9864,
      "obj": 1.0178
    }
  }
}
```

The script also accepts a list only when values are **genuine independent
beta-optimisation replicates**, such as repeated beta optimisation under
independently resampled calibration subsets or independent random seeds:

```json
{
  "condition_6": {
    "beta_values": {
      "cls":  [1.0412, 1.0389, 1.0441, 1.0405, 1.0428],
      "bbox": [0.9864, 0.9902, 0.9848, 0.9891, 0.9870],
      "obj":  [1.0178, 1.0155, 1.0190, 1.0168, 1.0183]
    }
  }
}
```

Do not populate these lists with six ingredient-model values unless each is
an independently re-optimised beta result. A beta is a property of the
merged M6 model, not an independent property of an ingredient model.

Run:

```bash
python -m yolof_soup.statistics.descriptive_beta_calibration
```

Output:

```text
results/descriptive_beta_calibration_results.json
```

## Running everything at once

After creating the bootstrap manifest and all four prediction-config files:

```bash
python -m yolof_soup.statistics.run_all_stats
```

The orchestrator runs all six analyses in sequence. Missing input files cause
that analysis to be marked `SKIPPED`; unexpected format errors are marked
`ERROR`. It writes a consolidated report to:

```text
results/statistical_summary.json
```

## Pre-run checklist

Before your final 5,000-replicate run, verify all items below:

- [ ] `results/ground_truth_heldout.json` contains only the held-out COCO
      val2017 image IDs, not the 953 calibration images.
- [ ] Every model prediction file contains only held-out image IDs.
- [ ] All models used identical COCO category IDs, score threshold, NMS, and
      evaluation configuration.
- [ ] The bootstrap smoke test completes with `B=100`.
- [ ] Direct full-set AP50:95 from `COCOeval` matches your stored experiment
      result for every model, subject only to rounding.
- [ ] `best_ingredient` and `best_condition_3to5` were selected on the
      calibration split, or you have labeled comparisons as exploratory.
- [ ] The final bootstrap manifest has `B=5000`, fixed `seed=42`, and is
      reused without modification by H1-H4.
- [ ] Results, prediction JSONs, ground truth, and the bootstrap manifest are
      archived with checksums for reproducibility.

## Superseded files (not deleted, not run by the orchestrator)

`h1_rq1_component_vs_global.py`, `h3_rm_anova_conditions2to5.py`,
`h4a_paired_ttest_M6_vs_M5.py`, `h4c_finetune_gain.py`,
`h2_rq2_lmc_barrier_anova.py`, and `h4b_beta_onesample_ttest.py` are earlier
versions retained for reference. They are not imported by
`run_all_stats.py`. Delete them only after confirming that the renamed
scripts produce the expected outputs.