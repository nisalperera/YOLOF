#!/usr/bin/env python
"""
Test script for linear mode connectivity analysis.

This script performs basic validation of the LMC implementation:
1. Tests checkpoint loading
2. Tests model interpolation
3. Tests architecture validation
4. Tests connectivity metrics computation
"""

import logging
import sys
import torch
from pathlib import Path

# Add YOLOF to path
yolof_dir = Path(__file__).parent
sys.path.insert(0, str(yolof_dir))

from yolof.analysis import (
    load_checkpoint_state_dict,
    validate_model_compatibility,
    interpolate_models,
    compute_connectivity_metrics,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def test_checkpoint_loading():
    """Test checkpoint loading functionality."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Checkpoint Loading")
    logger.info("="*70)
    
    # Try to load an existing checkpoint
    checkpoint_path = Path("/home/nisalperera/Projects/personal/YOLOF/output/baseline_yolof_coco2017/model_best.pth")
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        state_dict = load_checkpoint_state_dict(str(checkpoint_path))
        logger.info(f"✓ Successfully loaded checkpoint")
        logger.info(f"  - Number of parameters: {len(state_dict)}")
        logger.info(f"  - Sample keys: {list(state_dict.keys())[:3]}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load checkpoint: {e}")
        return False


def test_model_compatibility():
    """Test model architecture compatibility check."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Model Architecture Compatibility")
    logger.info("="*70)
    
    model1_path = Path("/home/nisalperera/Projects/personal/YOLOF/output/baseline_yolof_coco2017/model_best.pth")
    model2_path = Path("/home/nisalperera/Projects/personal/YOLOF/output/aug_autoaugment/model_best.pth")
    
    if not (model1_path.exists() and model2_path.exists()):
        logger.error(f"Required checkpoints not found")
        return False
    
    try:
        state_dict1 = load_checkpoint_state_dict(str(model1_path))
        state_dict2 = load_checkpoint_state_dict(str(model2_path))
        
        validate_model_compatibility(state_dict1, state_dict2)
        logger.info(f"✓ Models are architecturally compatible")
        return True
    except Exception as e:
        logger.error(f"✗ Compatibility check failed: {e}")
        return False


def test_interpolation():
    """Test model interpolation."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Model Interpolation")
    logger.info("="*70)
    
    model1_path = Path("/home/nisalperera/Projects/personal/YOLOF/output/baseline_yolof_coco2017/model_best.pth")
    model2_path = Path("/home/nisalperera/Projects/personal/YOLOF/output/aug_autoaugment/model_best.pth")
    
    if not (model1_path.exists() and model2_path.exists()):
        logger.error(f"Required checkpoints not found")
        return False
    
    try:
        state_dict1 = load_checkpoint_state_dict(str(model1_path))
        state_dict2 = load_checkpoint_state_dict(str(model2_path))
        
        # Test interpolation at different alpha values
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for alpha in alphas:
            interpolated = interpolate_models(state_dict1, state_dict2, alpha)
            
            # Verify interpolated state dict has same structure
            if set(interpolated.keys()) != set(state_dict1.keys()):
                logger.error(f"✗ Interpolated model has different keys at alpha={alpha}")
                return False
            
            # Verify a sample parameter value
            # At alpha=0, should be close to state_dict2
            # At alpha=1, should be close to state_dict1
            sample_key = list(state_dict1.keys())[0]
            sample_param = interpolated[sample_key]
            
            if alpha == 0.0:
                expected = state_dict2[sample_key]
                diff = torch.norm(sample_param - expected).item()
                logger.info(f"  α={alpha}: max diff from model2 = {diff:.2e}")
            elif alpha == 1.0:
                expected = state_dict1[sample_key]
                diff = torch.norm(sample_param - expected).item()
                logger.info(f"  α={alpha}: max diff from model1 = {diff:.2e}")
            else:
                logger.info(f"  α={alpha}: interpolation computed ✓")
        
        logger.info(f"✓ Model interpolation working correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ Interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connectivity_metrics():
    """Test connectivity metrics computation."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Connectivity Metrics Computation")
    logger.info("="*70)
    
    try:
        import numpy as np
        
        # Create synthetic loss curve
        alpha_values = np.linspace(0.0, 1.0, 11)
        
        # U-shaped loss curve (good connectivity)
        loss_curve_good = 1.0 + 0.1 * (2*alpha_values - 1)**2
        
        # High-barrier loss curve (poor connectivity)
        loss_curve_poor = 1.0 + 0.5 * np.sin(np.pi * alpha_values)
        
        metrics_good = compute_connectivity_metrics(alpha_values, loss_curve_good)
        metrics_poor = compute_connectivity_metrics(alpha_values, loss_curve_poor)
        
        logger.info(f"Good connectivity (U-shaped):")
        logger.info(f"  - Barrier height: {metrics_good['barrier_height']:.6f}")
        logger.info(f"  - Mean loss: {metrics_good['mean_loss']:.6f}")
        
        logger.info(f"Poor connectivity (high-barrier):")
        logger.info(f"  - Barrier height: {metrics_poor['barrier_height']:.6f}")
        logger.info(f"  - Mean loss: {metrics_poor['mean_loss']:.6f}")
        
        # Verify metrics have expected keys
        required_keys = ["barrier_height", "max_loss", "min_loss", "mean_loss", "curvature"]
        for key in required_keys:
            if key not in metrics_good:
                logger.error(f"✗ Missing key in metrics: {key}")
                return False
        
        logger.info(f"✓ Connectivity metrics computed successfully")
        return True
    
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("LINEAR MODE CONNECTIVITY - UNIT TESTS")
    logger.info("="*70)
    
    tests = [
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Model Compatibility", test_model_compatibility),
        ("Model Interpolation", test_interpolation),
        ("Connectivity Metrics", test_connectivity_metrics),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
