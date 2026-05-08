"""
utils/state_dict_utils.py
==========================
Safe state-dict assignment to YOLOF models.

Fix log:
  - assign_state_to_model: the `prefix` parameter is now actively used to
    strip a leading prefix from incoming keys before matching against the
    model's state-dict.  Previously the parameter existed but was never
    referenced, causing all soup keys to be silently ignored (extra_keys)
    whenever a prefix mismatch existed.
  - Shape mismatches now RAISE a RuntimeError instead of silently applying
    a corrupted tensor.
  - Added explicit guard: raises RuntimeError when zero keys matched,
    preventing the model from running with its original (pre-soup) weights.
"""

import logging
from typing import Dict

import torch

from yolof_soup.utils.inference import InferenceWrapper
from yolof_soup.utils.global_logger import get_logger

logger = get_logger(logging.DEBUG, add_file_handler=True)


def assign_state_to_model(
    model: torch.nn.Module | InferenceWrapper,
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "",
) -> None:
    """
    In-place replace model weights with those from *state_dict*.

    Args:
        model:      Target model (or InferenceWrapper).
        state_dict: Incoming weights.  Keys may optionally carry a leading
                    *prefix* that is absent from the model's own state-dict
                    (e.g. prefix="model." when a checkpoint was saved with
                    a DataParallel / DDP wrapper).
        prefix:     String to strip from incoming keys before matching.
                    Pass "" (default) when keys already match the model.

    Raises:
        RuntimeError: if zero keys matched after prefix stripping — this
                      almost always indicates a wrong prefix or wrong
                      checkpoint, and is far safer than silently running
                      with the original model weights.
        RuntimeError: if any matched key has a shape incompatible with the
                      model (would corrupt the tensor in-place).
    """
    model_state = model.state_dict()

    # Strip leading prefix from incoming keys
    normalised: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        norm_key = k[len(prefix):] if prefix and k.startswith(prefix) else k
        normalised[norm_key] = v

    incoming_keys = set(normalised.keys())
    model_keys    = set(model_state.keys())
    matched_keys  = incoming_keys & model_keys
    missing_keys  = model_keys - incoming_keys
    extra_keys    = incoming_keys - model_keys

    if not matched_keys:
        raise RuntimeError(
            f"assign_state_to_model: zero keys matched after prefix stripping "
            f"(prefix={prefix!r}).  "
            f"First 5 incoming keys: {sorted(incoming_keys)[:5]}  "
            f"First 5 model keys:    {sorted(model_keys)[:5]}"
        )

    if missing_keys:
        logger.warning(
            "assign_state_to_model: %d model keys not present in incoming "
            "state-dict (will keep existing values): %s …",
            len(missing_keys), sorted(missing_keys)[:5],
        )
    if extra_keys:
        logger.debug(
            "assign_state_to_model: %d extra incoming keys ignored: %s …",
            len(extra_keys), sorted(extra_keys)[:5],
        )

    # Validate shapes BEFORE applying — raise on mismatch
    shape_errors = []
    for key in matched_keys:
        if model_state[key].shape != normalised[key].shape:
            shape_errors.append(
                f"  '{key}': model={model_state[key].shape}, "
                f"incoming={normalised[key].shape}"
            )
    if shape_errors:
        raise RuntimeError(
            "assign_state_to_model: shape mismatch for "
            f"{len(shape_errors)} key(s):\n" + "\n".join(shape_errors)
        )

    # Apply matched keys
    model_state.update({k: normalised[k] for k in matched_keys})
    model.load_state_dict(model_state, strict=False)
    logger.info(
        "assign_state_to_model: applied %d / %d keys (prefix=%r).",
        len(matched_keys), len(model_keys), prefix,
    )
