"""
parameter_group_keys.py  (renamed from key_utils.py)
=====================================================
Maps Detectron2 module name prefixes to the five YOLOF parameter groups:
  backbone | encoder | cls_branch | reg_branch | object_pred

This mapping is the foundation of the tri-component merging formulation
described in thesis Section 3.3 (IV1).
"""
# Content preserved exactly from key_utils.py
from yolof_soup.utils.key_utils import *  # noqa: F401, F403
