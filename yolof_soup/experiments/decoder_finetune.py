"""
decoder_finetune.py  (renamed from head_finetuning.py)
=======================================================
Decoder-only post-merge fine-tuning runs (Steps 9-10, thesis Section 3.4):
  D1 - Init from Condition 2 (component uniform soup)
  D2 - Init from best of Conditions 3-6 (best learned soup)
  C3 - Full pipeline run on RTX 5090 (RQ4 principal result)

Backbone and encoder are frozen; only cls_branch, reg_branch,
and object_pred parameters are updated.
"""
# Content preserved exactly from head_finetuning.py
from yolof_soup.experiments.head_finetuning import *  # noqa: F401, F403
