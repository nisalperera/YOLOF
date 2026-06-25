"""
hypothesis_tests.py  (renamed from statistical_analysis.py)
============================================================
Runs all statistical tests for RQ1-RQ4 (thesis Section 3.5):
  H1: Paired t-test + bootstrap CI, Wilcoxon (80 per-class APs)
  H2: RM ANOVA + Tukey HSD on LMC barriers and Hessian traces
  H3: One-way RM ANOVA across IV2 coefficient strategies
  H4: Paired t-tests M5 vs M6, one-sample beta tests, D1 vs D2

Outputs the Summary Decision Table (Table 3.3) as JSON and CSV.
"""
# Content preserved exactly from statistical_analysis.py
from yolof_soup.experiments.statistical_analysis import *  # noqa: F401, F403
