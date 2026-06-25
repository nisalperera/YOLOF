"""
merge_coefficient_strategies.py  (renamed from coefficient_strategy_test.py)
=============================================================================
Implements and compares IV2 coefficient learning strategies:
  - Dirichlet random simplex search   (Condition C3)
  - Fisher-weighted coefficients      (Condition C4)
  - Jointly learned alpha+beta M5     (Condition C5)
  - Jointly learned alpha+beta M6     (Condition C6)

Note: despite the original '_test' suffix, this is NOT a pytest file.
"""
# Content preserved exactly from coefficient_strategy_test.py
from yolof_soup.experiments.coefficient_strategy_test import *  # noqa: F401, F403
