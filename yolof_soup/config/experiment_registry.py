"""
experiment_registry.py
======================
Canonical registry for thesis experiment runs and merge conditions.

This keeps run IDs (L1-L4, C1-C2, D1-D2, C3) separate from merge-condition
IDs (M1-M4) to avoid naming collisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExperimentRunSpec:
    """Metadata for one named experiment run."""

    run_id: str
    role: str
    run_type: str
    run_name: str
    changed_hyperparameter: str
    expected_gpu: str
    source_checkpoint_kind: str
    ingredient_index: Optional[int] = None


@dataclass(frozen=True)
class MergeConditionSpec:
    """Metadata for one merge condition."""

    condition_id: str
    name: str
    description: str


RUN_SPECS: Tuple[ExperimentRunSpec, ...] = (
    ExperimentRunSpec(
        run_id="L1",
        role="ingredient",
        run_type="full_finetune",
        run_name="finetune_thesis_L1",
        changed_hyperparameter="base_config_anchor",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=0,
    ),
    ExperimentRunSpec(
        run_id="L2",
        role="ingredient",
        run_type="full_finetune",
        run_name="finetune_thesis_L2",
        changed_hyperparameter="learning_rate",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=1,
    ),
    ExperimentRunSpec(
        run_id="L3",
        role="ingredient",
        run_type="full_finetune",
        run_name="finetune_thesis_L3",
        changed_hyperparameter="weight_decay",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=2,
    ),
    ExperimentRunSpec(
        run_id="L4",
        role="ingredient",
        run_type="full_finetune",
        run_name="finetune_thesis_L4",
        changed_hyperparameter="training_epochs",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=3,
    ),
    ExperimentRunSpec(
        run_id="C1",
        role="ingredient",
        run_type="full_finetune",
        run_name="finetune_thesis_C1",
        changed_hyperparameter="batch_size",
        expected_gpu="RTX 5090",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=4,
    ),
    ExperimentRunSpec(
        run_id="C2",
        role="ingredient",
        run_type="full_finetune",
        run_name="finetune_thesis_C2",
        changed_hyperparameter="lr_schedule",
        expected_gpu="RTX 5090",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=5,
    ),
    ExperimentRunSpec(
        run_id="D1",
        role="head_finetune",
        run_type="decoder_finetune",
        run_name="finetune_thesis_D1",
        changed_hyperparameter="merge_source_M2",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="merged_soup",
    ),
    ExperimentRunSpec(
        run_id="D2",
        role="head_finetune",
        run_type="decoder_finetune",
        run_name="finetune_thesis_D2",
        changed_hyperparameter="merge_source_best_of_M3_M4",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="merged_soup",
    ),
    ExperimentRunSpec(
        run_id="C3",
        role="final_pipeline",
        run_type="decoder_finetune",
        run_name="finetune_thesis_C3",
        changed_hyperparameter="final_pipeline_selection",
        expected_gpu="RTX 5090",
        source_checkpoint_kind="merged_soup",
    ),
)


MERGE_CONDITIONS: Tuple[MergeConditionSpec, ...] = (
    MergeConditionSpec(
        condition_id="M1",
        name="global_uniform",
        description="Global uniform soup over full model.",
    ),
    MergeConditionSpec(
        condition_id="M2",
        name="branch_uniform",
        description="Branch-partitioned uniform soup with separate cls/reg groups.",
    ),
    MergeConditionSpec(
        condition_id="M3",
        name="branch_dirichlet",
        description="Independent Dirichlet simplex search for cls/reg branch weights.",
    ),
    MergeConditionSpec(
        condition_id="M4",
        name="branch_fisher",
        description="Fisher-weighted branch soup using Hessian-based weights.",
    ),
)


def get_run_specs() -> Tuple[ExperimentRunSpec, ...]:
    return RUN_SPECS


def get_merge_conditions() -> Tuple[MergeConditionSpec, ...]:
    return MERGE_CONDITIONS


def list_runs_by_role(role: str) -> List[ExperimentRunSpec]:
    return [run for run in RUN_SPECS if run.role == role]


def ingredient_runs() -> List[ExperimentRunSpec]:
    runs = list_runs_by_role("ingredient")
    return sorted(runs, key=lambda r: int(r.ingredient_index or 0))


def ingredient_run_ids() -> List[str]:
    return [run.run_id for run in ingredient_runs()]


def validate_registry_specs(specs: Sequence[ExperimentRunSpec] = RUN_SPECS) -> List[str]:
    """Return a list of validation errors. Empty list means valid."""
    errors: List[str] = []

    run_ids = [run.run_id for run in specs]
    if len(set(run_ids)) != len(run_ids):
        errors.append("Run IDs must be unique.")

    condition_ids = [cond.condition_id for cond in MERGE_CONDITIONS]
    if len(set(condition_ids)) != len(condition_ids):
        errors.append("Merge condition IDs must be unique.")

    overlap = set(run_ids) & set(condition_ids)
    if overlap:
        errors.append(f"Run IDs and condition IDs overlap: {sorted(overlap)}")

    ingredient = [run for run in specs if run.role == "ingredient"]
    indexes = [run.ingredient_index for run in ingredient]
    if any(idx is None for idx in indexes):
        errors.append("All ingredient runs must define ingredient_index.")
    else:
        idx_vals = sorted(int(idx) for idx in indexes if idx is not None)
        expected = list(range(len(idx_vals)))
        if idx_vals != expected:
            errors.append(
                f"Ingredient indices must be contiguous 0..N-1. Got {idx_vals}."
            )

    return errors


def ingredient_checkpoint_paths(default_paths: Sequence[str]) -> List[str | Path]:
    """Map default decoder checkpoint paths to ingredient run order."""
    runs = ingredient_runs()
    if len(default_paths) < len(runs):
        raise ValueError(
            f"Not enough checkpoint paths: expected {len(runs)}, got {len(default_paths)}"
        )
    return [str(default_paths[int(run.ingredient_index or 0)]) for run in runs]


def ingredient_global_checkpoint_paths(default_paths: Sequence[str]) -> List[str | Path]:
    """Map default global checkpoint paths to ingredient run order."""
    runs = ingredient_runs()
    if len(default_paths) < len(runs):
        raise ValueError(
            f"Not enough global checkpoint paths: expected {len(runs)}, got {len(default_paths)}"
        )
    return [str(default_paths[int(run.ingredient_index or 0)]) for run in runs]


def build_experiment_manifest(
    decoder_paths: Sequence[str],
    global_paths: Sequence[str],
    checkpoint_dir: str,
) -> Dict[str, object]:
    """Build a JSON-serializable manifest with resolved run artifacts."""
    resolved_decoder = ingredient_checkpoint_paths(decoder_paths)
    resolved_global = ingredient_global_checkpoint_paths(global_paths)
    by_run: Dict[str, Dict[str, object]] = {}

    for i, run in enumerate(ingredient_runs()):
        by_run[run.run_id] = {
            **asdict(run),
            "decoder_checkpoint": resolved_decoder[i],
            "global_checkpoint": resolved_global[i],
        }

    checkpoint_root = Path(checkpoint_dir)
    for run in RUN_SPECS:
        if run.role == "ingredient":
            continue
        by_run[run.run_id] = {
            **asdict(run),
            "source_checkpoint": str(checkpoint_root / f"{run.run_id.lower()}_source.pth"),
        }

    return {
        "runs": by_run,
        "merge_conditions": [asdict(cond) for cond in MERGE_CONDITIONS],
        "ingredient_run_ids": ingredient_run_ids(),
    }
