import unittest
from pathlib import Path
import sys
import types


try:
    from yolof_soup.config.experiment_registry import (  # type: ignore
        build_experiment_manifest,
        get_merge_conditions,
        ingredient_checkpoint_paths,
        ingredient_run_ids,
        validate_registry_specs,
    )
except Exception:
    _REGISTRY_PATH = (
        Path(__file__).resolve().parents[1]
        / "yolof_soup"
        / "config"
        / "experiment_registry.py"
    )
    module_name = "experiment_registry_test_loader"
    _REGISTRY = types.ModuleType(module_name)
    _REGISTRY.__file__ = str(_REGISTRY_PATH)
    sys.modules[module_name] = _REGISTRY
    source = _REGISTRY_PATH.read_text(encoding="utf-8")
    code = compile(source, str(_REGISTRY_PATH), "exec")
    exec(code, _REGISTRY.__dict__)

    build_experiment_manifest = _REGISTRY.build_experiment_manifest
    get_merge_conditions = _REGISTRY.get_merge_conditions
    ingredient_checkpoint_paths = _REGISTRY.ingredient_checkpoint_paths
    ingredient_run_ids = _REGISTRY.ingredient_run_ids
    validate_registry_specs = _REGISTRY.validate_registry_specs


class TestExperimentRegistry(unittest.TestCase):
    def test_registry_validation_passes(self):
        errors = validate_registry_specs()
        self.assertEqual(errors, [])

    def test_condition_ids_do_not_overlap_runs(self):
        run_ids = set(ingredient_run_ids()) | {"D1", "D2", "C3"}
        cond_ids = {c.condition_id for c in get_merge_conditions()}
        self.assertTrue(run_ids.isdisjoint(cond_ids))

    def test_ingredient_path_mapping(self):
        default_paths = [f"decoder_{i}.pth" for i in range(6)]
        mapped = ingredient_checkpoint_paths(default_paths)
        self.assertEqual(mapped, default_paths)

    def test_manifest_contains_required_sections(self):
        dec_paths = [f"decoder_{i}.pth" for i in range(6)]
        glob_paths = [f"global_{i}.pth" for i in range(6)]
        manifest = build_experiment_manifest(dec_paths, glob_paths, "checkpoints")

        self.assertIn("runs", manifest)
        self.assertIn("merge_conditions", manifest)
        self.assertIn("ingredient_run_ids", manifest)

        for run_id in ["L1", "L2", "L3", "L4", "C1", "C2", "D1", "D2", "C3"]:
            self.assertIn(run_id, manifest["runs"])


if __name__ == "__main__":
    unittest.main()
