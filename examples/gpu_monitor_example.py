"""
GPU Memory Monitor Example

This script demonstrates various ways to use the GPU memory monitor.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor, start_gpu_monitor


def example_1_basic_usage():
    """Example 1: Basic usage with context manager."""
    print("=" * 70)
    print("Example 1: Basic Usage with Context Manager")
    print("=" * 70)
    
    # Monitor will start when entering the context and stop when exiting
    with GPUMemoryMonitor(
        log_file="logs/gpu_monitor_example1.log",
        interval=10,
        display=True,
        verbose=True
    ) as monitor:
        print("\nMonitor is running. Simulating workload for 30 seconds...")
        for i in range(3):
            print(f"  Working... ({i+1}/3)")
            time.sleep(10)
    
    print("\nMonitor stopped. Check logs/gpu_monitor_example1.log\n")


def example_2_manual_control():
    """Example 2: Manual start and stop control."""
    print("=" * 70)
    print("Example 2: Manual Start and Stop Control")
    print("=" * 70)
    
    # Create monitor
    monitor = GPUMemoryMonitor(
        log_file="logs/gpu_monitor_example2.log",
        interval=15,
        display=True,
        verbose=False
    )
    
    # Start monitoring
    monitor.start()
    print("\nMonitor started. Running for 45 seconds...")
    
    try:
        time.sleep(45)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        # Stop monitoring
        monitor.stop()
    
    print("\nMonitor stopped. Check logs/gpu_monitor_example2.log\n")


def example_3_convenient_function():
    """Example 3: Using the convenience function."""
    print("=" * 70)
    print("Example 3: Using Convenience Function")
    print("=" * 70)
    
    # Quick start with convenience function
    monitor = start_gpu_monitor(
        log_file="logs/gpu_monitor_example3.log",
        interval=20,
        display=True,
        verbose=True
    )
    
    print("\nMonitor started with convenience function. Running for 40 seconds...")
    
    try:
        time.sleep(40)
    finally:
        monitor.stop()
    
    print("\nMonitor stopped. Check logs/gpu_monitor_example3.log\n")


def example_4_polling_latest_stats():
    """Example 4: Polling for latest statistics."""
    print("=" * 70)
    print("Example 4: Polling Latest Statistics")
    print("=" * 70)
    
    monitor = GPUMemoryMonitor(
        log_file="logs/gpu_monitor_example4.log",
        interval=10,
        display=False,  # Don't print to stdout
        verbose=False
    )
    
    monitor.start()
    print("\nMonitor started (silent mode). Polling stats for 30 seconds...\n")
    
    try:
        for i in range(3):
            print(f"Iteration {i+1}:")
            time.sleep(10)
            
            # Get latest stats from the monitor
            latest_stats = monitor.get_latest_stats()
            
            if latest_stats and "gpus" in latest_stats:
                print(f"  Timestamp: {latest_stats['timestamp']}")
                for gpu in latest_stats["gpus"]:
                    print(f"  {gpu['name']}: {gpu['memory_used_mb']:.1f}MB / {gpu['memory_total_mb']:.1f}MB")
            else:
                print("  No GPU stats available")
            print()
    
    finally:
        monitor.stop()
    
    print("Check logs/gpu_monitor_example4.log for detailed logs\n")


def example_5_check_monitor_status():
    """Example 5: Checking monitor status."""
    print("=" * 70)
    print("Example 5: Checking Monitor Status")
    print("=" * 70)
    
    monitor = GPUMemoryMonitor(
        log_file="logs/gpu_monitor_example5.log",
        interval=5,
        display=True,
        verbose=False
    )
    
    print(f"\nBefore start - Is running: {monitor.is_running()}")
    
    monitor.start()
    print(f"After start - Is running: {monitor.is_running()}")
    
    time.sleep(15)
    
    monitor.stop()
    print(f"After stop - Is running: {monitor.is_running()}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPU Memory Monitor Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_monitor_example.py --example 1
  python gpu_monitor_example.py --example all
  python gpu_monitor_example.py --example 3 --verbose
        """
    )
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which example to run (1-5), or 'all' for all examples"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples sequentially"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    examples = {
        1: example_1_basic_usage,
        2: example_2_manual_control,
        3: example_3_convenient_function,
        4: example_4_polling_latest_stats,
        5: example_5_check_monitor_status,
    }
    
    if args.all or (hasattr(args, 'all') and args.all):
        for num in [1, 2, 3, 4, 5]:
            examples[num]()
            print("\n" + "=" * 70 + "\n")
    else:
        examples[args.example]()
