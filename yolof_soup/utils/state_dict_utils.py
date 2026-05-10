import logging

from typing import Dict

import torch

from yolof_soup.utils.inference import EvaluateModel
from yolof_soup.utils.global_logger import get_logger

logger = get_logger(logging.DEBUG, add_file_handler=True)  # Use global logger for diagnostics

def assign_state_to_model(
    model: torch.nn.Module | EvaluateModel,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """In-place replace model's decoder weights with those from state_dict."""
    

    if isinstance(model, EvaluateModel):
        model_state = model.model.state_dict()
    else:
        model_state = model.state_dict()
    
    # Find missing keys (only log if there are issues)
    incoming_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())
    missing_keys = model_keys - incoming_keys
    extra_keys = incoming_keys - model_keys
    
    if missing_keys:
        logger.warning(f"assign_state_to_model: {len(missing_keys)} keys missing from incoming state dict")
        logger.debug(f"  Missing keys: {sorted(missing_keys)[:10]}")
    if extra_keys:
        logger.debug(f"assign_state_to_model: {len(extra_keys)} extra keys in incoming state (will be ignored)")
    
    # Validate incoming keys
    shape_mismatches = []
    for key, value in state_dict.items():
        if key not in model_state:
            # This is OK - will be skipped in update
            pass
        elif model_state[key].shape != value.shape:
            logger.error(
                f"Shape mismatch for '{key}': "
                f"model={model_state[key].shape}, new={value.shape}"
            )
    
    # Apply update
    model_state.update(state_dict)
    if isinstance(model, EvaluateModel):
        model.model.load_state_dict(model_state, strict=True)  # strict=False to allow missing keys
    else:
        model.load_state_dict(model_state, strict=True)  # strict=True to ensure all keys are present