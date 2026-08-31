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

    id: str
    role: str
    run_type: str
    run_name: str
    changed_hyperparameter: str
    expected_gpu: str
    source_checkpoint_kind: str
    checkpoint_path: Path
    eval_json_path: Path
    ingredient_index: Optional[int] = None


@dataclass(frozen=True)
class MergeConditionSpec:
    """Metadata for one merge condition."""

    id: str
    name: str
    description: str
    eval_json_path: Path
    checkpoint_path: Optional[Path] = None


RUN_SPECS: Tuple[ExperimentRunSpec, ...] = (
    ExperimentRunSpec(
        id="L1",
        role="ingredient",
        run_type="full_finetune",
        run_name="L1",
        changed_hyperparameter="base_config_anchor",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=0,
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/ingridients-refined/L1/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/L1/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="L2",
        role="ingredient",
        run_type="full_finetune",
        run_name="L2",
        changed_hyperparameter="learning_rate",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=1,
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/ingridients-refined/L2/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/L2/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="L3",
        role="ingredient",
        run_type="full_finetune",
        run_name="L3",
        changed_hyperparameter="weight_decay",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=2,
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/ingridients-refined/L3/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/L3/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="L4",
        role="ingredient",
        run_type="full_finetune",
        run_name="L4",
        changed_hyperparameter="training_epochs",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=3,
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/ingridients-refined/L4/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/L4/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="R1",
        role="ingredient",
        run_type="full_finetune",
        run_name="R1",
        changed_hyperparameter="batch_size",
        expected_gpu="RTX 5090",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=4,
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/ingridients-refined/R1/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/R1/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="R2",
        role="ingredient",
        run_type="full_finetune",
        run_name="R2",
        changed_hyperparameter="lr_schedule",
        expected_gpu="RTX 5090",
        source_checkpoint_kind="pretrained_base",
        ingredient_index=5,
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/ingridients-refined/R2/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/R2/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="D1",
        role="decoder_finetune",
        run_type="decoder_finetune",
        run_name="D1",
        changed_hyperparameter="merge_source_M2",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="merged_soup",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/decoder-finetune/D1/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/D1/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="D2",
        role="decoder_finetune",
        run_type="decoder_finetune",
        run_name="D2",
        changed_hyperparameter="merge_source_best_of_M3_M4",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="merged_soup",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/decoder-finetune/D2/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/D2/coco_instances_results.json")
    ),
    ExperimentRunSpec(
        id="C3",
        role="decoder_finetune",
        run_type="decoder_finetune",
        run_name="C3",
        changed_hyperparameter="final_pipeline_selection",
        expected_gpu="RTX 5070 Ti",
        source_checkpoint_kind="merged_soup",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/soup_exps/decoder-finetune/C3/model_best.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/C3/coco_instances_results.json")
    ),
)


MERGE_CONDITIONS: Tuple[MergeConditionSpec, ...] = (
    MergeConditionSpec(
        id="M1",
        name="global_uniform",
        description="Global uniform soup over full model.",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/global_uniform_soup.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/M1/coco_instances_results.json")
    ),
    MergeConditionSpec(
        id="M2",
        name="branch_uniform",
        description="Branch-partitioned uniform soup with separate cls/reg groups.",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/branch_uniform_soup.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/M2/coco_instances_results.json")
    ),
    MergeConditionSpec(
        id="M3",
        name="branch_dirichlet",
        description="Independent Dirichlet simplex search for cls/reg branch weights.",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/dirichlet_soup.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/M3/coco_instances_results.json")
    ),
    MergeConditionSpec(
        id="M4",
        name="branch_fisher",
        description="Fisher-weighted branch soup using Hessian-based weights.",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/fisher_soup.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/M4/coco_instances_results.json")
    ),
    MergeConditionSpec(
        id="M5",
        name="uniform_learned",
        description="Jointly optimises mixing coefficients α ∈ ℝᵏ and temperature β ∈ ℝ",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/uniform_learned_soup.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/M5/coco_instances_results.json")
    ),
    MergeConditionSpec(
        id="M6",
        name="learned_tri_head",
        description="Maintains three independent (α, β) pairs for the three decoder prediction " \
            "sub-heads, while backbone and encoder are fixed under uniform averaging",
        checkpoint_path=Path("/home/nisalperera/YOLOF/output/learned_tri_head_soup.pth"),
        eval_json_path=Path("/home/nisalperera/YOLOF/results/inference/final_eval/M6/coco_instances_results.json")
    )
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
    return [run.id for run in ingredient_runs()]


def validate_registry_specs(specs: Sequence[ExperimentRunSpec] = RUN_SPECS) -> List[str]:
    """Return a list of validation errors. Empty list means valid."""
    errors: List[str] = []

    run_ids = [run.id for run in specs]
    if len(set(run_ids)) != len(run_ids):
        errors.append("Run IDs must be unique.")

    condition_ids = [cond.id for cond in MERGE_CONDITIONS]
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
        by_run[run.id] = {
            **asdict(run),
            "decoder_checkpoint": resolved_decoder[i],
            "global_checkpoint": resolved_global[i],
        }

    checkpoint_root = Path(checkpoint_dir)
    for run in RUN_SPECS:
        if run.role == "ingredient":
            continue
        by_run[run.id] = {
            **asdict(run),
            "source_checkpoint": str(checkpoint_root / f"{run.id.lower()}_source.pth"),
        }

    return {
        "runs": by_run,
        "merge_conditions": [asdict(cond) for cond in MERGE_CONDITIONS],
        "ingredient_run_ids": ingredient_run_ids(),
    }
