"""
checkpoint_preregistration.py  (renamed from preregistration.py)
=================================================================
Step 8 of the data collection procedure (thesis Section 3.4):
pre-registers the source soup checkpoints for D1 and D2 before
any decoder-only fine-tuning begins, preventing post-hoc
selection bias in IV3 experiments.
"""
# Content preserved exactly from preregistration.py
from yolof_soup.experiments.preregistration import *  # noqa: F401, F403
