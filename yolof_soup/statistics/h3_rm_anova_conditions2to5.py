
import numpy as np
import pandas as pd
from scipy import stats

# Replace with your actual 80 per-class AP arrays (one AP value per COCO category)
# Order must be identical across all four arrays (same category order)
cond2_per_class_ap = np.array([...])  # length 80, Condition 2 (component uniform soup)
cond3_per_class_ap = np.array([...])  # length 80, Condition 3 (Dirichlet)
cond4_per_class_ap = np.array([...])  # length 80, Condition 4 (Fisher-weighted)
m5_per_class_ap     = np.array([...])  # length 80, M5 (shared alpha/beta)

for arr in (cond2_per_class_ap, cond3_per_class_ap, cond4_per_class_ap, m5_per_class_ap):
    assert len(arr) == 80

n_classes = 80
category_ids = np.arange(n_classes)

# Build long-format dataframe: subject = category, within-factor = condition
df = pd.DataFrame({
    "subject": np.tile(category_ids, 4),
    "condition": np.repeat(["Condition2", "Condition3", "Condition4", "M5"], n_classes),
    "ap": np.concatenate([cond2_per_class_ap, cond3_per_class_ap, cond4_per_class_ap, m5_per_class_ap])
})

# ---- Option A: pingouin (preferred, cleaner API, includes GG correction + sphericity test) ----
try:
    import pingouin as pg

    aov = pg.rm_anova(dv="ap", within="condition", subject="subject", data=df, detailed=True)
    print("=== RM-ANOVA (pingouin) ===")
    print(aov)

    sph = pg.sphericity(df, dv="ap", within="condition", subject="subject")
    print("\nSphericity test (Mauchly):")
    print(sph)

    posthoc = pg.pairwise_tests(dv="ap", within="condition", subject="subject", data=df, padjust="bonf")
    print("\n=== Tukey/Bonferroni-corrected post-hoc pairwise contrasts ===")
    print(posthoc)

except ImportError:
    print("pingouin not installed; falling back to statsmodels AnovaRM (Option B below)")

    # ---- Option B: statsmodels AnovaRM (no built-in sphericity test or GG correction) ----
    from statsmodels.stats.anova import AnovaRM

    aovrm = AnovaRM(df, depvar="ap", subject="subject", within=["condition"]).fit()
    print("=== RM-ANOVA (statsmodels) ===")
    print(aovrm.summary())

    # Manual Greenhouse-Geisser check requires raw covariance matrix; statsmodels does not
    # provide Mauchly's test directly. Use pingouin for GG correction and sphericity, or
    # compute Mauchly's W manually if pingouin cannot be installed.

    # Tukey HSD post-hoc (treats data as if independent -- less appropriate for RM design,
    # but usable as an approximate post-hoc if pingouin is unavailable)
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(endog=df["ap"], groups=df["condition"], alpha=0.05)
    print("\n=== Tukey HSD post-hoc (approximate, treats conditions as independent groups) ===")
    print(tukey)
