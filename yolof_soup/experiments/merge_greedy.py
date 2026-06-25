"""
merge_greedy.py  (renamed from greedy_soup.py)
===============================================
Standard greedy model soup following Wortsman et al. (2022):
iteratively add ingredient models only if they improve val2017 mAP.
Provides an additional comparison baseline alongside C1-C6.
"""
# Content preserved exactly from greedy_soup.py
from yolof_soup.experiments.greedy_soup import *  # noqa: F401, F403
