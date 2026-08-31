# Statistics update notes (final thesis alignment)

## Cross-chapter numbering inconsistency found in the final draft

While reading the final thesis PDF, I found that Chapter 1 Sec.1.5 and
Chapter 4's Preliminary Note use a RENUMBERED, sequential H1-H4 scheme,
but Chapter 3 Sec.3.5 still describes tests under an OLDER labeling
scheme. Both chapters cannot be correct simultaneously; the code in this
directory now follows Ch.1/Ch.4 as authoritative, because:

1. Ch.1 Sec.1.5's introductory sentence ("final hypothesis... H1, H3,
   H4a, and H4c... H2 and H4b... descriptively only") is a STALE leftover
   from an earlier draft -- immediately afterward, the SAME section lists
   exactly four hypotheses, sequentially renumbered H1, H2, H3, H4 (not
   H1/H3/H4a/H4c).
2. Ch.4's Preliminary Note independently and explicitly confirms this
   renumbering: "static learning effect addresses both H1 and RQ1... 
   strategy equivalence across Conditions 2 to 6 addresses H2 and RQ2...
   M6 versus M5 is addressed in H3 and RQ3... full pipeline superiority
   addresses H4 and RQ4... the per-component loss landscape geometry
   analysis and temperature calibration deviation analysis, both of which
   were FORMERLY H2 and H4b respectively IN THE EARLIER DRAFT, are
   exclusively maintained as descriptive context."

### Final mapping used in this code

| Final label | Content | Old label (Ch.3 Sec.3.5 still uses this) |
|---|---|---|
| H1 / RQ1 | M6 vs Condition 1, and vs best individual ingredient | H1 / RQ1 (unchanged) |
| H2 / RQ2 | Strategy equivalence across Conditions 2, 3, 4, M5, M6 | was H3/RQ3 ("Worthy Strategy Contrast") |
| H3 / RQ3 | M6 vs M5 | was H4a ("Granularity Contrast") |
| H4 / RQ4 | C3 vs best individual ingredient AND vs published baseline | was H4c ("Pipeline cap check") |
| *(descriptive, no label)* | LMC barriers (Bcls vs Breg) | was H2 ("Geometric Curvature check") |
| *(descriptive, no label)* | Beta temperature calibration | was H4b ("Calibration Deviation Check") |

### ACTION NEEDED IN THE THESIS TEXT (not just code)

Chapter 3 Sec.3.5 needs to be rewritten to:
1. Use the final H1-H4 labels above, not the old ones.
2. Describe the paired image-level bootstrap methodology (see below)
   instead of the one-sample/paired t-test, repeated-measures ANOVA with
   Greenhouse-Geisser correction, Tukey HSD, and Bonferroni-adjusted
   one-sample t-test design currently described there. That design treats
   80 COCO categories as independent repeated measures / independent
   observations, which is invalid (see rationale below). I can draft this
   replacement Section 3.5 text on request.

## Why per-class-AP inference was replaced with paired image-level bootstrap

COCO category AP values are correlated components of one aggregate mAP
estimate (they share images, scene composition, and detector failure
modes), so treating 80 (or 70) per-class AP values as independent
repeated measures for a t-test, Wilcoxon test, RM-ANOVA, or Kruskal-Wallis
test is pseudo-replication. The confirmatory H1-H4 scripts instead
resample held-out COCO evaluation IMAGES with replacement (B=5,000
replicates, seed=42), retaining all ground truth and predictions per
sampled image, and recompute AP50:95 on each resample for both compared
models (paired design). See `bootstrap_core.py` for the full
implementation and docstring.

## File-by-file summary of this update

| File | Status |
|---|---|
| `h1_rq1_static_learning_effect.py` | New; supersedes `h1_rq1_component_vs_global.py`. Trimmed to exactly Ha1's two comparisons (M6 vs C1, M6 vs best individual); the old "Comparison B" (Condition 2 vs best learned) moved to H2. |
| `h2_rq2_strategy_equivalence.py` | New; supersedes `h3_rm_anova_conditions2to5.py`. Same planned-contrast bootstrap family, relabeled H2/RQ2. |
| `h3_rq3_m6_vs_m5.py` | New; supersedes `h4a_paired_ttest_M6_vs_M5.py`. Content unchanged, relabeled H3/RQ3. |
| `h4_rq4_full_pipeline_superiority.py` | New; supersedes `h4c_finetune_gain.py`. Confirmatory test narrowed to exactly Ha4 (C3 vs best individual, headline); published-baseline comparison and D1/D2/C3 fine-tuning-gain pairs reframed as supporting descriptive evidence, not the formal decision. |
| `descriptive_lmc_barrier_geometry.py` | New; supersedes `h2_rq2_lmc_barrier_anova.py`. Same computation; formal "REJECT/FAIL TO REJECT H02" decision language removed since this is descriptive-only in the final draft. Added caveat: the 15 pairs are not fully independent (share 6 underlying models). |
| `descriptive_beta_calibration.py` | New; supersedes `h4b_beta_onesample_ttest.py`. Same descriptive-only content, relabeled to drop the H4b tag. |
| `run_all_stats.py` | Rewritten to import the six new modules, added a preflight check for the bootstrap manifest, and a reminder to delete superseded files. |
| `README.md` | New; per-test run instructions for all six analyses. |

## Superseded files (NOT deleted)

`h1_rq1_component_vs_global.py`, `h3_rm_anova_conditions2to5.py`,
`h4a_paired_ttest_M6_vs_M5.py`, `h4c_finetune_gain.py`,
`h2_rq2_lmc_barrier_anova.py`, `h4b_beta_onesample_ttest.py` remain in the
repo for now. They are no longer imported by `run_all_stats.py`. Delete
them once you've verified the renamed versions produce expected output --
I won't delete files without your explicit confirmation.

## Still outstanding

- Chapter 3 Sec.3.5 rewrite (labels + bootstrap methodology description) -- offer available on request.
- `bootstrap_core.py` and `generate_coco_bootstrap_manifest.py` are unchanged from the previous update; no changes needed for this realignment.
