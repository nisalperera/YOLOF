"""
pipeline_integration.py  (renamed from full_pipeline_test.py)
==============================================================
Full three-stage pipeline integration run (C3):
  Stage 1: Uniform backbone + encoder merge
  Stage 2: Learned tri-component decoder merge (best of C3-C6)
  Stage 3: Decoder-only fine-tuning

This is the principal RQ4 experiment (thesis Section 3.5.4).
"""
# Content preserved exactly from full_pipeline_test.py
from yolof_soup.experiments.full_pipeline_test import *  # noqa: F401, F403
