"""
merge_conditions.py  (renamed from soup_construction.py)
=========================================================
Implements all six merging conditions (C1-C6) from thesis Section 3.3.2:

  C1 - Global uniform soup         (baseline, IV1 absent)
  C2 - Component uniform soup      (IV1 present, IV2 uniform)
  C3 - Component Dirichlet search  (IV1 present, IV2 Dirichlet random)
  C4 - Component Fisher-weighted   (IV1 present, IV2 Fisher)
  C5 - M5: Jointly learned alpha+beta, shared pair
  C6 - M6: Jointly learned alpha+beta, independent per-component pairs

The tri-component parameter grouping (cls_branch, reg_branch, object_pred)
corresponds to IV1 in the conceptual framework (Chapter 2, Section 2.9).
"""
# Content preserved exactly from soup_construction.py
from yolof_soup.experiments.soup_construction import *  # noqa: F401, F403
