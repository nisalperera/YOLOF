# Statistics update: paired image-level bootstrap replaces per-class inference

## What changed

| File | Status | Change |
|---|---|---|
| `bootstrap_core.py` | New | Shared paired image-level bootstrap engine (resampling unit = one held-out COCO image, not a category). |
| `h1_rq1_component_vs_global.py` | Rewritten | Comparisons A/B/C now use paired image-level bootstrap instead of `ttest_rel` / `wilcoxon` / class-resampled bootstrap. Holm-Bonferroni applied across the 3 comparisons. |
| `h3_rm_anova_conditions2to5.py` | Rewritten | Test 1 replaced with a limited planned pairwise contrast family (Condition 2 vs 3/4/5/6, Condition 3 vs 4) via image-level bootstrap, Holm-adjusted. Test 3 (coefficient magnitude, N=6 ingredient models) kept largely as-is since that sampling unit is legitimate. M6 vs M5 intentionally excluded here to avoid double-counting with H4a. |
| `h4a_paired_ttest_M6_vs_M5.py` | Rewritten | Single confirmatory M6 vs M5 image-level bootstrap comparison. |
| `h4b_beta_onesample_ttest.py` | Rewritten | Converted to a **descriptive report**. A single learned β scalar has no valid replicate sampling distribution; the one-sample t-test and ANOVA over ingredient-model coefficients were not statistically justified. If genuine independent β-optimisation replicates exist (different seeds/resampled calibration subsets), a bootstrap CI function is provided and can be used instead. |
| `h4c_finetune_gain.py` | Rewritten | Test 4 (D1/D2/C3 gain) uses paired image-level bootstrap per pair, Holm-adjusted. Test 5's Kruskal-Wallis/Mann-Whitney over per-class gain arrays (invalid: non-independent groups) replaced with a direct D2 vs C3 image-level bootstrap comparison. C3 vs the published 37.7 AP baseline is now explicitly **descriptive only**. |

## NOT changed (needs your manual review)

- **`h2_rq2_lmc_barrier_anova.py`** -- not modified. Its sampling unit (15 model pairs for LMC barrier comparisons) is not COCO categories, so it may not have the same pseudo-replication problem, but I could not retrieve its current content through the available tooling to verify. Please review it against the same principle: the resampling/test unit must be something independent-ish (ingredient-model pairs, images), never COCO categories or per-class AP values.
- **`run_all_stats.py`** -- not modified. It likely still imports the old per-script APIs. Update it to:
  1. Run `generate_coco_bootstrap_manifest.py` once (fixed seed=42) to produce `bootstrap_image_ids.npy` before any H1/H3/H4a/H4c script runs.
  2. Call each rewritten script with `--bootstrap-image-ids` pointing at that same file, so all comparisons share one paired bootstrap design.
  3. If you want a single Holm correction across the *entire* confirmatory family (H1's 3 comparisons + H3's 5 + H4a's 1 + H4c's 3-4), collect all raw p-values across scripts and apply `holm_adjust()` once centrally, instead of per-script, to avoid inconsistent correction scope. The current per-script Holm application is a reasonable default but is scoped per-family (H1, H3, H4c), matching the compact hypothesis set in Chapter 1 (H1, H3, H4a, H4c).

## Required one-time setup

Before running any of the rewritten scripts, generate the shared bootstrap draws once:

```bash
python yolof_soup/statistics/generate_coco_bootstrap_manifest.py \
  --ground-truth results/ground_truth_heldout.json \
  --predictions results/predictions/*.json \
  --iterations 5000 \
  --seed 42 \
  --output-dir results/bootstrap_manifest
```

All of `h1_rq1_component_vs_global.py`, `h3_rm_anova_conditions2to5.py`, `h4a_paired_ttest_M6_vs_M5.py`, and `h4c_finetune_gain.py` expect `results/bootstrap_manifest/bootstrap_image_ids.npy` to already exist and share the same seed/replicate count, so that every pairwise contrast in the thesis uses the identical paired resampling scheme.

## Config files you need to create

- `configs/h1_predictions.json`
- `configs/h3_predictions.json`
- `configs/h4a_predictions.json`
- `configs/h4c_predictions.json`

Each maps condition/model names to already-generated COCO-format prediction JSON files (see docstring at the top of each script for the exact expected keys and an example).

## Thesis wording to update (Chapter 3, Section 3.5)

Replace the RM-ANOVA/Tukey/paired-t-test/Wilcoxon/Kruskal-Wallis description with:

> Primary inferential comparisons were conducted using paired non-parametric bootstrap resampling at the image level. The held-out COCO validation images formed the resampling units, and images were sampled with replacement for B = 5,000 replicates (seed = 42). All ground-truth annotations and stored detections belonging to each sampled image occurrence were retained, with duplicate draws assigned distinct synthetic image identifiers. For each replicate, COCO AP50:95 was recalculated for both compared models on the identical resample, and the paired AP difference was recorded. Percentile 95% confidence intervals and one-sided bootstrap p-values were estimated from the resulting distribution, with Holm-Bonferroni adjustment applied within each predefined confirmatory comparison family. Per-class AP and AR were reported descriptively and were not treated as independent observations.
