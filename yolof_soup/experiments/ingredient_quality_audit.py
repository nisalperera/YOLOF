"""
ingredient_quality_audit.py  (renamed from quality_audit.py)
=============================================================
Step 5 of the data collection procedure (thesis Section 3.4):
evaluates all N=6 ingredient checkpoints on COCO val2017 and
flags any model more than 3 pp below the pool maximum mAP
for potential exclusion (greedy inclusion principle).
"""
# Content preserved exactly from quality_audit.py
from yolof_soup.experiments.quality_audit import *  # noqa: F401, F403
