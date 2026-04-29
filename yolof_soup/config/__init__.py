"""
configs/__init__.py
Re-export the most commonly imported symbols so experiment scripts can do:
    from configs import build_eval_cfg, DEVICE, RESULTS_DIR
instead of the longer:
    from configs.experiment_config import ...
"""

from .experiment_config import (
    # Environment
    NUM_GPUS,
    DEVICE,
    AMP_ENABLED,
    # Directories
    PROJECT_ROOT,
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LOG_DIR,
    PHASE1_OUTPUT_DIR,
    PHASE2_OUTPUT_DIR,
    # Dataset identifiers
    TRAIN_DATASET,
    SELECTION_DATASET,
    EVAL_DATASET,
    VOC_DATASET,
    SELECTION_ANN,
    EVAL_ANN,
    VOC_ANN,
    # Checkpoint paths
    N_INGREDIENTS,
    PRETRAINED_WEIGHTS,
    BACKBONE_ENC_CKPT,
    DECODER_CKPT_PATHS,
    GLOBAL_CKPT_PATHS,
    BASELINE_CKPT,
    LEARNED_SOUP_CKPT,
    GREEDY_SOUP_CKPT,
    UNIFORM_SOUP_CKPT,
    GLOBAL_SOUP_CKPT,
    # Soup-construction
    LAMBDA_GRID,
    MAX_CD_PASSES,
    CONVERGE_TOL,
    SELECTION_EVAL_MAX_IMGS,
    # Loss landscape
    LMC_ALPHA_STEPS,
    SAM_RHO,
    SHARPNESS_STEPS,
    LANDSCAPE_EVAL_MAX_IMGS,
    # Statistics
    ALPHA_SIGNIFICANCE,
    N_BOOTSTRAP,
    COHENS_D_THRESHOLD,
    SPEARMAN_R_THRESHOLD,
    # CfgNode factories
    build_eval_cfg,
)

__all__ = [
    "NUM_GPUS", "DEVICE", "AMP_ENABLED",
    "PROJECT_ROOT", "DATA_DIR", "CHECKPOINT_DIR", "RESULTS_DIR",
    "LOG_DIR", "PHASE1_OUTPUT_DIR", "PHASE2_OUTPUT_DIR",
    "TRAIN_DATASET", "SELECTION_DATASET", "EVAL_DATASET", "VOC_DATASET",
    "SELECTION_ANN", "EVAL_ANN", "VOC_ANN",
    "N_INGREDIENTS", "PRETRAINED_WEIGHTS",
    "BACKBONE_ENC_CKPT", "DECODER_CKPT_PATHS", "GLOBAL_CKPT_PATHS",
    "BASELINE_CKPT", "LEARNED_SOUP_CKPT", "GREEDY_SOUP_CKPT",
    "UNIFORM_SOUP_CKPT", "GLOBAL_SOUP_CKPT",
    "BASE_LR", "WEIGHT_DECAY", "MOMENTUM", "CLIP_GRAD_NORM",
    "BATCH_SIZE_PER_GPU", "SEEDS", "AUG_SCHEDULES",
    "PHASE1_MAX_ITER", "PHASE2_MAX_ITER",
    "LAMBDA_GRID", "MAX_CD_PASSES", "CONVERGE_TOL", "SELECTION_EVAL_MAX_IMGS",
    "LMC_ALPHA_STEPS", "SAM_RHO", "SHARPNESS_STEPS", "LANDSCAPE_EVAL_MAX_IMGS",
    "ALPHA_SIGNIFICANCE", "N_BOOTSTRAP", "COHENS_D_THRESHOLD",
    "SPEARMAN_R_THRESHOLD",
    "build_phase1_cfg", "build_phase2_cfg", "build_eval_cfg",
]