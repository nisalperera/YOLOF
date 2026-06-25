"""
GPU Monitor Verification Script

This script verifies that the GPU monitor is properly set up and working.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 70)
    print("Checking Dependencies")
    print("=" * 70)
    
    dependencies = {
        'pynvml': 'nvidia-ml-py3',
        'psutil': 'psutil',
    }
    
    missing = []
    
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name:20s} installed")
        except ImportError:
            print(f"✗ {module_name:20s} NOT installed (pip install {package_name})")
            missing.append(package_name)
    
    return missing


def check_gpu_availability():
    """Check if GPUs are available."""
    print("\n" + "=" * 70)
    print("Checking GPU Availability")
    print("=" * 70)
    
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        print(f"✓ NVIDIA CUDA available")
        print(f"✓ Number of GPUs detected: {gpu_count}")
        
        if gpu_count > 0:
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / (1024**3)
                print(f"  GPU {i}: {name} ({total_gb:.1f}GB)")
            return True
        else:
            print("⚠ No GPUs detected")
            return False
    
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False


def test_import():
    """Test if the GPU monitor can be imported."""
    print("\n" + "=" * 70)
    print("Testing GPU Monitor Import")
    print("=" * 70)
    
    try:
        from yolof_soup.utils.gpu_memory_usage import (
            GPUMemoryMonitor,
            start_gpu_monitor
        )
        print("✓ Successfully imported GPUMemoryMonitor")
        print("✓ Successfully imported start_gpu_monitor")
        return True
    except Exception as e:
        print(f"✗ Failed to import GPU monitor: {e}")
        return False


def test_monitor_creation():
    """Test creating and configuring a monitor."""
    print("\n" + "=" * 70)
    print("Testing Monitor Creation")
    print("=" * 70)
    
    try:
        from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor
        
        # Create monitor with test settings
        monitor = GPUMemoryMonitor(
            log_file="test_logs/test_monitor.log",
            interval=5,
            display=False,
            verbose=False
        )
        
        print("✓ Monitor instance created")
        print(f"  Log file: {monitor.log_file}")
        print(f"  Interval: {monitor.interval}s")
        print(f"  Display: {monitor.display}")
        print(f"  Verbose: {monitor.verbose}")
        print(f"  Is running: {monitor.is_running()}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create monitor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitor_lifecycle():
    """Test starting and stopping the monitor."""
    print("\n" + "=" * 70)
    print("Testing Monitor Lifecycle")
    print("=" * 70)
    
    try:
        import time
        from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor
        
        monitor = GPUMemoryMonitor(
            log_file="test_logs/lifecycle_test.log",
            interval=2,
            display=False,
            verbose=False
        )
        
        print("Before start:")
        print(f"  Is running: {monitor.is_running()}")
        
        # Start monitor
        monitor.start()
        print("\nAfter start:")
        print(f"  Is running: {monitor.is_running()}")
        
        # Let it run for a few seconds
        print("  Waiting 5 seconds...")
        time.sleep(5)
        
        # Check measurement count
        print(f"  Measurements taken: {monitor._measurement_count}")
        
        # Get latest stats
        stats = monitor.get_latest_stats()
        if stats:
            print(f"  Latest stats timestamp: {stats['timestamp']}")
            if "gpus" in stats:
                print(f"  Number of GPUs in stats: {len(stats['gpus'])}")
        
        # Stop monitor
        monitor.stop()
        print("\nAfter stop:")
        print(f"  Is running: {monitor.is_running()}")
        print(f"  Total measurements: {monitor._measurement_count}")
        
        # Check log file
        if os.path.exists(monitor.log_file):
            with open(monitor.log_file, 'r') as f:
                lines = f.readlines()
            print(f"  Log file size: {len(lines)} lines")
            if len(lines) > 0:
                print(f"  First log line: {lines[0].strip()}")
                if len(lines) > 1:
                    print(f"  Last log line: {lines[-1].strip()}")
        
        return True
    
    except Exception as e:
        print(f"✗ Monitor lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_manager():
    """Test using monitor as context manager."""
    print("\n" + "=" * 70)
    print("Testing Context Manager")
    print("=" * 70)
    
    try:
        import time
        from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor
        
        with GPUMemoryMonitor(
            log_file="test_logs/context_manager_test.log",
            interval=2,
            display=False,
            verbose=False
        ) as monitor:
            print("Inside context manager:")
            print(f"  Is running: {monitor.is_running()}")
            time.sleep(3)
            print(f"  Measurements: {monitor._measurement_count}")
        
        print("Outside context manager:")
        print(f"  Is running: {monitor.is_running()}")
        
        return True
    
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_checks():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "GPU MONITOR VERIFICATION SUITE" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Create test logs directory
    os.makedirs("test_logs", exist_ok=True)
    
    results = {}
    
    # Run checks
    missing_deps = check_dependencies()
    results['dependencies'] = len(missing_deps) == 0
    
    results['gpu_available'] = check_gpu_availability()
    results['import'] = test_import()
    results['creation'] = test_monitor_creation()
    results['lifecycle'] = test_monitor_lifecycle()
    results['context_manager'] = test_context_manager()
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    checks = {
        'Dependencies': results['dependencies'],
        'GPU Available': results['gpu_available'],
        'Monitor Import': results['import'],
        'Monitor Creation': results['creation'],
        'Monitor Lifecycle': results['lifecycle'],
        'Context Manager': results['context_manager'],
    }
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:30s}: {status}")
    
    print("\n" + "=" * 70)
    
    if all(checks.values()):
        print("✓ All checks passed! GPU Monitor is ready to use.")
        print("\nNext steps:")
        print("  1. Read the GPU_MONITOR_GUIDE.md for usage instructions")
        print("  2. Run examples: python examples/gpu_monitor_example.py")
        print("  3. Integrate into your training script")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        if missing_deps:
            print(f"\nMissing dependencies: {', '.join(missing_deps)}")
            print(f"Install with: pip install {' '.join(missing_deps)}")
        return 1


if __name__ == "__main__":
    exit_code = run_all_checks()
    sys.exit(exit_code)
