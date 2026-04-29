"""
utils/__init__.py
==================
Flat re-export of every public symbol from the four utility modules.

    from utils import load_state, compute_coco_map, wilcoxon_one_tailed
    from utils import get_decoder_keys, compute_anchor, apply_subhead_lambdas
"""

from utils.checkpoint_utils import (
    load_state,
    load_states,
    load_metadata,
    save_checkpoint,
    save_ingredients,
)

from utils.key_utils import (
    get_decoder_keys,
    get_backbone_encoder_keys,
    split_decoder_subheads,
    extract_subdict,
    merge_subdicts,
    compute_anchor,
    compute_task_vectors,
    apply_uniform_lambdas,
    apply_subhead_lambdas,
)

from utils.eval_utils import (
    build_eval_dataloader,
    compute_coco_map,
    get_map,
    quick_loss,
)

from utils.stats_utils import (
    wilcoxon_one_tailed,
    cohens_d,
    mann_whitney_u_test,
    wilcoxon_paired,
    spearman_r,
    compare_diversity_gain,
    compare_domain_gains,
    bootstrap_ci,
)

__all__ = [
    "load_state", "load_states", "load_metadata",
    "save_checkpoint", "save_ingredients",
    "get_decoder_keys", "get_backbone_encoder_keys",
    "split_decoder_subheads", "extract_subdict", "merge_subdicts",
    "compute_anchor", "compute_task_vectors",
    "apply_uniform_lambdas", "apply_subhead_lambdas",
    "build_eval_dataloader", "compute_coco_map", "get_map", "quick_loss",
    "wilcoxon_one_tailed", "cohens_d",
    "mann_whitney_u_test", "wilcoxon_paired",
    "spearman_r", "compare_diversity_gain",
    "compare_domain_gains", "bootstrap_ci",
]