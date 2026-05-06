"""
Full pipeline integration test for YOLOF Soup experiments.

This module orchestrates the complete experimental workflow:
  Phase 3: Soup construction (M1-M4)
  Phase 4: Loss landscape analysis (LMC barriers, Hessian traces)
  Phase 5: Head fine-tuning (D1, D2, C3)
  RQ3: Coefficient strategy tests
  RQ4: Full pipeline and per-category analysis

Each phase is executed sequentially with validation checks to ensure
data dependencies are satisfied. Results are aggregated into a
comprehensive integration report.

Usage:
  python -m yolof_soup.experiments.integration_test
  python -m yolof_soup.experiments.integration_test --validate-only
  python -m yolof_soup.experiments.integration_test --resume-from phase5
"""

import os
import sys
import json
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from yolof_soup.config.experiment_config import RESULTS_DIR, DATA_DIR
from yolof_soup.config.experiment_registry import INGREDIENT_SPECS, MERGE_CONDITIONS


def validate_input_data() -> Dict[str, bool]:
    """
    Validate that all required input data exists.
    
    Returns:
        Dict with validation results for each required component
    """
    print("\n" + "=" * 80)
    print("INPUT VALIDATION")
    print("=" * 80)
    
    validation_results = {}
    
    # Check ingredient checkpoints
    print("\nChecking ingredient model checkpoints...")
    for spec_id, spec in INGREDIENT_SPECS.items():
        checkpoint_path = Path(spec.get('checkpoint_path', ''))
        exists = checkpoint_path.exists()
        validation_results[f'ingredient_{spec_id}'] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {spec_id}: {checkpoint_path}")
    
    # Check COCO dataset
    print("\nChecking COCO dataset...")
    coco_train = Path(DATA_DIR) / "coco" / "train2017"
    coco_val = Path(DATA_DIR) / "coco" / "val2017"
    train_exists = coco_train.exists()
    val_exists = coco_val.exists()
    validation_results['coco_train'] = train_exists
    validation_results['coco_val'] = val_exists
    print(f"  {'✓' if train_exists else '✗'} COCO train2017: {coco_train}")
    print(f"  {'✓' if val_exists else '✗'} COCO val2017: {coco_val}")
    
    # Check results directory
    print("\nChecking results directory...")
    results_path = Path(RESULTS_DIR)
    results_exists = results_path.exists()
    validation_results['results_dir'] = results_exists
    print(f"  {'✓' if results_exists else '✗'} Results directory: {results_path}")
    
    return validation_results


def run_phase_3() -> bool:
    """Run Phase 3: Soup Construction."""
    print("\n" + "=" * 80)
    print("PHASE 3: SOUP CONSTRUCTION (M1-M4)")
    print("=" * 80)
    
    try:
        from yolof_soup.experiments.soup_construction import run_phase3
        
        print("\nRunning soup construction with 4 conditions...")
        results = run_phase3()
        
        # Validate output
        phase3_results = Path(RESULTS_DIR) / "phase3_soup_results.json"
        if phase3_results.exists():
            with open(phase3_results) as f:
                data = json.load(f)
            print(f"✓ Phase 3 complete: {len(data)} condition results saved")
            return True
        else:
            print("✗ Phase 3 results file not found")
            return False
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
        traceback.print_exc()
        return False


def run_phase_4() -> bool:
    """Run Phase 4: Loss Landscape Analysis."""
    print("\n" + "=" * 80)
    print("PHASE 4: LOSS LANDSCAPE ANALYSIS")
    print("=" * 80)
    
    try:
        from yolof_soup.experiments.loss_landscape import run_phase4
        
        print("\nRunning loss landscape analysis (LMC barriers, Hessian traces)...")
        results = run_phase4()
        
        # Validate outputs
        lmc_results = Path(RESULTS_DIR) / "phase4_lmc_barriers.json"
        hessian_results = Path(RESULTS_DIR) / "phase4_hessian_traces.json"
        stats_results = Path(RESULTS_DIR) / "phase4_statistical_tests.json"
        
        outputs_exist = all([lmc_results.exists(), hessian_results.exists(), stats_results.exists()])
        
        if outputs_exist:
            print(f"✓ Phase 4 complete:")
            print(f"  - LMC barriers: {lmc_results}")
            print(f"  - Hessian traces: {hessian_results}")
            print(f"  - Statistical tests: {stats_results}")
            return True
        else:
            print("✗ Phase 4 results files not found")
            return False
    except Exception as e:
        print(f"✗ Phase 4 failed: {e}")
        traceback.print_exc()
        return False


def run_phase_5() -> bool:
    """Run Phase 5: Head Fine-Tuning."""
    print("\n" + "=" * 80)
    print("PHASE 5: HEAD FINE-TUNING (D1, D2, C3)")
    print("=" * 80)
    
    try:
        from yolof_soup.experiments.head_finetuning import run_phase5
        
        print("\nRunning head fine-tuning for D1, D2, C3 variants...")
        results = run_phase5()
        
        # Validate outputs
        phase5_results = Path(RESULTS_DIR) / "phase5_finetuning_results.json"
        d1_checkpoint = Path(RESULTS_DIR) / "d1_finetuned.pth"
        d2_checkpoint = Path(RESULTS_DIR) / "d2_finetuned.pth"
        c3_checkpoint = Path(RESULTS_DIR) / "c3_pipeline.pth"
        
        checkpoints_exist = all([d1_checkpoint.exists(), d2_checkpoint.exists(), c3_checkpoint.exists()])
        results_exist = phase5_results.exists()
        
        if results_exist and checkpoints_exist:
            print(f"✓ Phase 5 complete:")
            print(f"  - Results: {phase5_results}")
            print(f"  - D1 checkpoint: {d1_checkpoint}")
            print(f"  - D2 checkpoint: {d2_checkpoint}")
            print(f"  - C3 checkpoint: {c3_checkpoint}")
            return True
        else:
            print("✗ Phase 5 outputs incomplete")
            if not results_exist:
                print(f"  - Missing: {phase5_results}")
            if not checkpoints_exist:
                print(f"  - Missing checkpoints")
            return False
    except Exception as e:
        print(f"✗ Phase 5 failed: {e}")
        traceback.print_exc()
        return False


def run_rq3() -> bool:
    """Run RQ3: Coefficient Strategy Tests."""
    print("\n" + "=" * 80)
    print("RQ3: COEFFICIENT STRATEGY TESTS")
    print("=" * 80)
    
    try:
        from yolof_soup.experiments.coefficient_strategy_test import run_rq3
        
        print("\nRunning coefficient strategy comparison and moderation analysis...")
        results = run_rq3()
        
        # Validate output
        rq3_results = Path(RESULTS_DIR) / "rq3_test_results.json"
        if rq3_results.exists():
            with open(rq3_results) as f:
                data = json.load(f)
            print(f"✓ RQ3 complete: {len(data)} test results saved")
            return True
        else:
            print("✗ RQ3 results file not found")
            return False
    except Exception as e:
        print(f"✗ RQ3 failed: {e}")
        traceback.print_exc()
        return False


def run_rq4() -> bool:
    """Run RQ4: Full Pipeline and Per-Category Analysis."""
    print("\n" + "=" * 80)
    print("RQ4: FULL PIPELINE & PER-CATEGORY ANALYSIS")
    print("=" * 80)
    
    try:
        from yolof_soup.experiments.full_pipeline_test import run_rq4
        
        print("\nRunning full pipeline evaluation and per-category analysis...")
        results = run_rq4()
        
        # Validate output
        rq4_results = Path(RESULTS_DIR) / "rq4_test_results.json"
        if rq4_results.exists():
            with open(rq4_results) as f:
                data = json.load(f)
            print(f"✓ RQ4 complete: {len(data)} analysis results saved")
            return True
        else:
            print("✗ RQ4 results file not found")
            return False
    except Exception as e:
        print(f"✗ RQ4 failed: {e}")
        traceback.print_exc()
        return False


def generate_integration_report(results: Dict[str, bool]) -> str:
    """
    Generate a comprehensive integration test report.
    
    Args:
        results: Dictionary of phase results
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("\n" + "=" * 80)
    report.append("INTEGRATION TEST REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().isoformat()}\n")
    
    # Summary
    total_phases = len(results)
    passed_phases = sum(1 for v in results.values() if v)
    passed_percentage = (passed_phases / total_phases * 100) if total_phases > 0 else 0
    
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Phases: {total_phases}")
    report.append(f"Passed: {passed_phases}")
    report.append(f"Failed: {total_phases - passed_phases}")
    report.append(f"Success Rate: {passed_percentage:.1f}%\n")
    
    # Phase results
    report.append("PHASE RESULTS")
    report.append("-" * 80)
    phase_order = ["Phase 3", "Phase 4", "Phase 5", "RQ3", "RQ4"]
    for phase in phase_order:
        if phase in results:
            status = "✓ PASS" if results[phase] else "✗ FAIL"
            report.append(f"  {status}: {phase}")
    
    report.append("\n")
    report.append("GENERATED ARTIFACTS")
    report.append("-" * 80)
    
    artifacts = {
        "Phase 3 Results": RESULTS_DIR / "phase3_soup_results.json",
        "Phase 4 LMC Barriers": RESULTS_DIR / "phase4_lmc_barriers.json",
        "Phase 4 Hessian Traces": RESULTS_DIR / "phase4_hessian_traces.json",
        "Phase 4 Statistical Tests": RESULTS_DIR / "phase4_statistical_tests.json",
        "Phase 5 Results": RESULTS_DIR / "phase5_finetuning_results.json",
        "Phase 5 D1 Checkpoint": RESULTS_DIR / "d1_finetuned.pth",
        "Phase 5 D2 Checkpoint": RESULTS_DIR / "d2_finetuned.pth",
        "Phase 5 C3 Checkpoint": RESULTS_DIR / "c3_pipeline.pth",
        "RQ3 Results": RESULTS_DIR / "rq3_test_results.json",
        "RQ4 Results": RESULTS_DIR / "rq4_test_results.json",
    }
    
    for artifact_name, artifact_path in artifacts.items():
        exists = Path(artifact_path).exists()
        status = "✓" if exists else "✗"
        report.append(f"  {status} {artifact_name}")
    
    report.append("\n")
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)
    
    if passed_percentage == 100:
        report.append("✓ All phases completed successfully!")
        report.append("  Next steps:")
        report.append("  1. Review results in the results/ directory")
        report.append("  2. Check individual phase JSON outputs for detailed metrics")
        report.append("  3. Validate per-category analysis in RQ4 results")
    else:
        report.append("✗ Some phases failed. Review error messages above.")
        report.append("  Common issues:")
        report.append("  - Missing ingredient checkpoints (Phase 1/2 outputs)")
        report.append("  - COCO dataset not downloaded or missing splits")
        report.append("  - GPU memory issues (reduce batch size in config)")
        report.append("  - Missing dependencies (pip install -r requirements.txt)")
    
    report.append("\n" + "=" * 80)
    report.append("END REPORT")
    report.append("=" * 80 + "\n")
    
    return "\n".join(report)


def validate_only() -> bool:
    """Run validation checks without executing experiments."""
    validation = validate_input_data()
    
    required_items = {
        'results_dir': 'Results directory',
        'coco_train': 'COCO train dataset',
        'coco_val': 'COCO val dataset',
    }
    
    # Check if any ingredient is available
    ingredient_items = [k for k in validation.keys() if k.startswith('ingredient_')]
    any_ingredient_exists = any(validation.get(k, False) for k in ingredient_items)
    
    all_valid = all(validation.get(k, False) for k in required_items.keys()) and any_ingredient_exists
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✓ All validations passed. Ready to run integration test.")
    else:
        print("✗ Validation failed. Missing required components:")
        for key, name in required_items.items():
            if not validation.get(key, False):
                print(f"  - {name}")
        if not any_ingredient_exists:
            print("  - At least one ingredient checkpoint")
    print("=" * 80 + "\n")
    
    return all_valid


def main(validate_only_flag: bool = False, resume_from: Optional[str] = None):
    """
    Execute the full integration test pipeline.
    
    Args:
        validate_only_flag: If True, only validate inputs without running experiments
        resume_from: Phase to resume from (e.g., 'phase5', 'rq3')
    """
    print("\n" + "=" * 80)
    print("YOLOF SOUP - FULL INTEGRATION TEST")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}\n")
    
    # Validate inputs
    if not validate_only():
        if not validate_only_flag:
            print("\n⚠ Warning: Some inputs missing. Attempting to continue...")
    
    if validate_only_flag:
        print("Validation-only mode complete. Exiting.")
        return
    
    # Phase execution
    results = {}
    phases = [
        ("Phase 3", run_phase_3),
        ("Phase 4", run_phase_4),
        ("Phase 5", run_phase_5),
        ("RQ3", run_rq3),
        ("RQ4", run_rq4),
    ]
    
    # Determine start index
    start_idx = 0
    if resume_from:
        for idx, (phase_name, _) in enumerate(phases):
            if resume_from.lower() in phase_name.lower():
                start_idx = idx
                break
    
    if start_idx > 0:
        print(f"\nResuming from {phases[start_idx][0]}...\n")
    
    # Execute phases
    for phase_name, phase_func in phases[start_idx:]:
        results[phase_name] = phase_func()
    
    # Generate report
    report = generate_integration_report(results)
    print(report)
    
    # Save report to file
    report_path = Path(RESULTS_DIR) / "integration_test_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}\n")
    
    # Exit code based on success
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    print(f"Exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Full pipeline integration test for YOLOF Soup experiments'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without running experiments'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from specified phase (phase3, phase4, phase5, rq3, rq4)'
    )
    
    args = parser.parse_args()
    main(validate_only_flag=args.validate_only, resume_from=args.resume_from)
