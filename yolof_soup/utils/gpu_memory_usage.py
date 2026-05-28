"""
GPU Memory Usage Monitor

This module provides a GPU memory monitor that runs as a separate thread,
collecting and logging GPU memory statistics every minute.

Usage:
    # Start the monitor
    monitor = GPUMemoryMonitor(log_file="gpu_stats.log", interval=60)
    monitor.start()
    
    # Stop the monitor
    monitor.stop()
    
    # With context manager
    with GPUMemoryMonitor(log_file="gpu_stats.log", interval=60) as monitor:
        # Your code here
        pass
"""

import os
import sys
import threading
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import queue

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[WARNING] pynvml not installed. Install with: pip install nvidia-ml-py3")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not installed. Install with: pip install psutil")


class GPUMemoryMonitor:
    """
    A GPU memory monitor that runs in a separate thread and logs metrics every minute.
    
    Attributes:
        log_file (str): Path to the log file for GPU statistics
        interval (int): Time interval in seconds between measurements (default: 60)
        display (bool): Whether to display stats to stdout (default: True)
        verbose (bool): Whether to show detailed metrics (default: False)
    """
    
    def __init__(
        self,
        log_file: str = "gpu_memory_stats.log",
        interval: int = 60,
        display: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the GPU Memory Monitor.
        
        Args:
            log_file: Path to the log file
            interval: Interval in seconds between measurements
            display: Whether to print stats to stdout
            verbose: Whether to show detailed information
        """
        self.log_file = log_file
        self.interval = interval
        self.display = display
        self.verbose = verbose
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._data_queue: queue.Queue = queue.Queue()
        self._measurement_count = 0
        self._start_time = None
        
        # Setup logger
        self._setup_logger()
        
        # Initialize NVIDIA utilities if available
        self._init_nvidia()
        
    def _setup_logger(self):
        """Setup logging for the GPU monitor."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger("GPUMemoryMonitor")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional)
        if self.display:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                "[GPU Monitor] %(asctime)s | %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def _init_nvidia(self):
        """Initialize NVIDIA GPU utilities."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.logger.info(f"Detected {self.gpu_count} GPU(s)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVIDIA utilities: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary containing GPU statistics
        """
        stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gpus": []
        }
        
        if not PYNVML_AVAILABLE or self.gpu_count == 0:
            stats["error"] = "GPU monitoring not available"
            return stats
        
        try:
            for gpu_id in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get GPU utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = None
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                except:
                    temp = None
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                except:
                    power = None
                
                gpu_stats = {
                    "gpu_id": gpu_id,
                    "name": pynvml.nvmlDeviceGetName(handle).decode() if isinstance(
                        pynvml.nvmlDeviceGetName(handle), bytes
                    ) else pynvml.nvmlDeviceGetName(handle),
                    "memory_used_mb": mem_info.used / (1024 ** 2),
                    "memory_total_mb": mem_info.total / (1024 ** 2),
                    "memory_used_percent": (mem_info.used / mem_info.total) * 100,
                    "memory_free_mb": mem_info.free / (1024 ** 2),
                }
                
                if gpu_util is not None:
                    gpu_stats["utilization_percent"] = gpu_util
                if temp is not None:
                    gpu_stats["temperature_c"] = temp
                if power is not None:
                    gpu_stats["power_usage_w"] = power
                
                stats["gpus"].append(gpu_stats)
        
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def format_stats(self, stats: Dict[str, Any]) -> str:
        """
        Format statistics for display and logging.
        
        Args:
            stats: Dictionary of GPU statistics
            
        Returns:
            Formatted string
        """
        if "error" in stats:
            return f"[{stats['timestamp']}] Error: {stats['error']}"
        
        lines = [f"[{stats['timestamp']}] GPU Memory Report (Measurement #{self._measurement_count})"]
        
        for gpu in stats["gpus"]:
            name = gpu["name"]
            used = gpu["memory_used_mb"]
            total = gpu["memory_total_mb"]
            percent = gpu["memory_used_percent"]
            
            line = f"  {name} | Memory: {used:.1f}MB / {total:.1f}MB ({percent:.1f}%)"
            
            if self.verbose:
                free = gpu["memory_free_mb"]
                line += f" | Free: {free:.1f}MB"
                
                if "utilization_percent" in gpu:
                    line += f" | Util: {gpu['utilization_percent']:.1f}%"
                if "temperature_c" in gpu:
                    line += f" | Temp: {gpu['temperature_c']:.1f}°C"
                if "power_usage_w" in gpu:
                    line += f" | Power: {gpu['power_usage_w']:.1f}W"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in the separate thread."""
        self._start_time = time.time()
        self.logger.info("GPU Memory Monitor started")
        
        while not self._stop_event.is_set():
            try:
                self._measurement_count += 1
                
                # Collect statistics
                stats = self.get_gpu_stats()
                
                # Format and display
                formatted_stats = self.format_stats(stats)
                
                # Log to file and stdout
                if "error" not in stats:
                    for line in formatted_stats.split("\n"):
                        self.logger.info(line)
                else:
                    self.logger.warning(formatted_stats)
                
                # Add to queue for external access
                self._data_queue.put(stats)
                
                # Wait for the next interval
                self._stop_event.wait(self.interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                self._stop_event.wait(self.interval)
        
        elapsed_time = time.time() - self._start_time
        self.logger.info(
            f"GPU Memory Monitor stopped (ran for {elapsed_time:.1f}s, "
            f"{self._measurement_count} measurements)"
        )
    
    def start(self):
        """Start the GPU memory monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning("Monitor is already running")
            return
        
        self._stop_event.clear()
        self._measurement_count = 0
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=False,
            name="GPUMemoryMonitor"
        )
        self._thread.start()
    
    def stop(self):
        """Stop the GPU memory monitoring thread."""
        if self._thread is None:
            self.logger.warning("Monitor is not running")
            return
        
        self._stop_event.set()
        self._thread.join(timeout=5)
        
        if self._thread.is_alive():
            self.logger.error("Monitor thread did not stop gracefully")
        else:
            self.logger.info("Monitor stopped successfully")
    
    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest GPU statistics from the queue.
        
        Returns:
            Latest stats dictionary or None if queue is empty
        """
        try:
            # Get all items from queue, keeping only the latest
            latest = None
            while True:
                latest = self._data_queue.get_nowait()
        except queue.Empty:
            pass
        
        return latest
    
    def get_latest_memory_usage(self):
        """
        Get the latest GPU memory usage in percentage.
        
        Returns:
            List of memory usage for each GPU or None if stats are unavailable
        """
        stats = self.get_latest_stats()
        if stats is None:
            return []
        if "error" in stats:
            return None
        
        return [
            gpu["memory_used_percent"] for gpu in stats.get("gpus", [])
        ]
    
    def is_running(self) -> bool:
        """Check if the monitor is currently running."""
        return self._thread is not None and self._thread.is_alive()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Convenience function for quick setup
def start_gpu_monitor(
    log_file: str = "gpu_memory_stats.log",
    interval: int = 60,
    display: bool = True,
    verbose: bool = False
) -> GPUMemoryMonitor:
    """
    Convenience function to quickly start a GPU memory monitor.
    
    Args:
        log_file: Path to the log file
        interval: Interval in seconds between measurements
        display: Whether to print stats to stdout
        verbose: Whether to show detailed information
        
    Returns:
        Running GPUMemoryMonitor instance
    """
    monitor = GPUMemoryMonitor(
        log_file=log_file,
        interval=interval,
        display=display,
        verbose=verbose
    )
    monitor.start()
    return monitor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPU Memory Usage Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_memory_usage.py --log-file gpu_stats.log --interval 60
  python gpu_memory_usage.py --display --verbose
  python gpu_memory_usage.py --interval 30 --no-display
        """
    )
    parser.add_argument(
        "--log-file",
        default="gpu_memory_stats.log",
        help="Path to the log file (default: gpu_memory_stats.log)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=True,
        help="Display stats to stdout (default: True)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable stdout display"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metrics"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Run for a specific duration in seconds (default: run indefinitely)"
    )
    
    args = parser.parse_args()
    
    display = not args.no_display if args.no_display else args.display
    
    # Create and start monitor
    monitor = GPUMemoryMonitor(
        log_file=args.log_file,
        interval=args.interval,
        display=display,
        verbose=args.verbose
    )
    
    try:
        monitor.start()
        
        if args.duration:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("GPU Memory Monitor running. Press Ctrl+C to stop.")
            while monitor.is_running():
                print(monitor.get_latest_stats())
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        monitor.stop()
