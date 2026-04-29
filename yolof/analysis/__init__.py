"""
Analysis utilities for YOLOF models: mode connectivity, loss landscape, model soup.
"""

from .mode_connectivity import (
    interpolate_models,
    interpolate_state_dict,
    load_checkpoint_state_dict,
    validate_model_compatibility,
    evaluate_loss_on_dataset,
    compute_connectivity_metrics,
)
from .loss_landscape import LossLandscape
from .model_soup import build_soup, save_soup, BACKBONE_ENCODER_PREFIXES

__all__ = [
    # mode connectivity
    "interpolate_models",
    "interpolate_state_dict",
    "load_checkpoint_state_dict",
    "validate_model_compatibility",
    "evaluate_loss_on_dataset",
    "compute_connectivity_metrics",
    # loss landscape
    "LossLandscape",
    # model soup
    "build_soup",
    "save_soup",
    "BACKBONE_ENCODER_PREFIXES",
]
