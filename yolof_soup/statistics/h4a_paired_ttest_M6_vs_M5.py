import json
import numpy as np
from scipy import stats

# Replace with your actual 80 per-class AP arrays (one AP value per COCO category)
# Order must be identical between the two arrays (same category order)
with open('phase3_soup_results', 'r') as f:
    soup_results = json.load(f)
    m6_per_class_ap = np.array(soup_results['condition_6']["per_class_ap"])  # length 80
    m5_per_class_ap = np.array(soup_results['condition_5']["per_class_ap"])  # length 80

m6_per_class_ap = np.array([...])  # length 80
m5_per_class_ap = np.array([...])  # length 80

assert len(m6_per_class_ap) == 80 and len(m5_per_class_ap) == 80

# Paired t-test
t_stat, p_value = stats.ttest_rel(m6_per_class_ap, m5_per_class_ap)
df = len(m6_per_class_ap) - 1

# Cohen's d for paired samples (using SD of the differences)
diff = m6_per_class_ap - m5_per_class_ap
cohens_d = diff.mean() / diff.std(ddof=1)

# Bootstrap 95% CI on the mean difference
rng = np.random.default_rng(42)
n_boot = 10000
boot_means = np.empty(n_boot)
n = len(diff)
for i in range(n_boot):
    sample = rng.choice(diff, size=n, replace=True)
    boot_means[i] = sample.mean()
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

print(f"t({df}) = {t_stat:.4f}, p = {p_value:.4f}")
print(f"Mean difference (M6 - M5) = {diff.mean():.4f} pp")
print(f"Cohen's d = {cohens_d:.4f}")
print(f"95% bootstrap CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

# Wilcoxon signed-rank as non-parametric robustness check (per Ch.3 3.5.1/3.5.4)
w_stat, w_p = stats.wilcoxon(m6_per_class_ap, m5_per_class_ap)
print(f"Wilcoxon: W = {w_stat:.4f}, p = {w_p:.4f}")
