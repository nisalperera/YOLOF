"""
GPU Memory Monitor Integration Example

This script demonstrates how to integrate the GPU memory monitor
with a typical training loop.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor


class TrainingWithGPUMonitor:
    """
    Example training class that includes GPU memory monitoring.
    """
    
    def __init__(self, model_name="YOLOF", num_epochs=10):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.gpu_monitor = None
    
    def train(self):
        """Main training loop with GPU monitoring."""
        
        # Initialize GPU monitor
        log_dir = "logs/training_gpu_stats"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{self.model_name}_gpu_stats.log")
        
        # Create and start GPU monitor
        self.gpu_monitor = GPUMemoryMonitor(
            log_file=log_file,
            interval=60,  # Log every minute
            display=True,
            verbose=True
        )
        
        self.gpu_monitor.start()
        print(f"GPU Monitor started. Logging to: {log_file}")
        
        try:
            for epoch in range(self.num_epochs):
                print(f"\n{'='*70}")
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
                print(f"{'='*70}")
                
                # Simulate training steps
                num_steps = 10
                for step in range(num_steps):
                    # Simulate training step
                    self._train_step(epoch, step, num_steps)
                    
                    # Optional: Get current GPU stats
                    if step % 5 == 0:
                        stats = self.gpu_monitor.get_latest_stats()
                        if stats and "gpus" in stats:
                            gpu = stats["gpus"][0]
                            print(f"  Step {step}/{num_steps} - GPU Memory: "
                                  f"{gpu['memory_used_mb']:.1f}MB / "
                                  f"{gpu['memory_total_mb']:.1f}MB "
                                  f"({gpu['memory_used_percent']:.1f}%)")
                
                # Simulate validation
                print(f"Validation...")
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
        
        finally:
            # Stop GPU monitor
            if self.gpu_monitor:
                self.gpu_monitor.stop()
                print(f"\nGPU Monitor stopped. Stats saved to: {log_file}")
    
    def _train_step(self, epoch, step, num_steps):
        """Simulate a training step."""
        progress = (step + 1) / num_steps
        print(f"  Training... [{step+1}/{num_steps}] ({progress*100:.1f}%)")
        time.sleep(1)  # Simulate computation


def example_training_integration():
    """Run an example training with GPU monitoring."""
    print("GPU Monitor - Training Integration Example")
    print("=" * 70)
    
    trainer = TrainingWithGPUMonitor(
        model_name="YOLOF_Demo",
        num_epochs=3
    )
    
    trainer.train()


def example_multi_process_training():
    """
    Example showing how to use GPU monitor with multiprocessing.
    
    Note: In real scenarios, you might run this in a separate process.
    """
    print("\n" + "=" * 70)
    print("GPU Monitor - Multi-Process Example")
    print("=" * 70)
    
    # Create monitor for multi-process training
    monitor = GPUMemoryMonitor(
        log_file="logs/training_gpu_stats/multiprocess_gpu_stats.log",
        interval=30,  # Check every 30 seconds
        display=True,
        verbose=False
    )
    
    monitor.start()
    print("\nGPU Monitor started for multi-process training")
    
    try:
        # Simulate multi-process training
        print("Simulating multi-process training for 60 seconds...")
        for i in range(6):
            print(f"  Training iteration {i+1}/6...")
            time.sleep(10)
    
    finally:
        monitor.stop()


def example_monitor_with_callbacks():
    """
    Example showing how to monitor GPU memory and trigger callbacks
    based on memory thresholds.
    """
    print("\n" + "=" * 70)
    print("GPU Monitor - Callback Example")
    print("=" * 70)
    
    class MemoryThresholdMonitor(GPUMemoryMonitor):
        """Extended monitor with memory threshold callbacks."""
        
        def __init__(self, *args, memory_threshold=80.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory_threshold = memory_threshold
        
        def _monitor_loop(self):
            """Override monitor loop to add threshold checking."""
            self._start_time = time.time()
            self.logger.info("GPU Memory Monitor with threshold started")
            
            while not self._stop_event.is_set():
                try:
                    self._measurement_count += 1
                    stats = self.get_gpu_stats()
                    
                    # Check memory threshold
                    if "gpus" in stats:
                        for gpu in stats["gpus"]:
                            mem_percent = gpu["memory_used_percent"]
                            
                            if mem_percent > self.memory_threshold:
                                self.logger.warning(
                                    f"Memory threshold exceeded on {gpu['name']}: "
                                    f"{mem_percent:.1f}% > {self.memory_threshold}%"
                                )
                                self._on_memory_threshold_exceeded(gpu)
                    
                    # Log stats
                    formatted_stats = self.format_stats(stats)
                    if "error" not in stats:
                        for line in formatted_stats.split("\n"):
                            self.logger.info(line)
                    
                    self._data_queue.put(stats)
                    self._stop_event.wait(self.interval)
                
                except Exception as e:
                    self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                    self._stop_event.wait(self.interval)
        
        def _on_memory_threshold_exceeded(self, gpu_stats):
            """Callback when memory threshold is exceeded."""
            print(f"⚠️  ALERT: GPU {gpu_stats['gpu_id']} memory threshold exceeded!")
            # Could trigger garbage collection, model checkpointing, etc.
    
    monitor = MemoryThresholdMonitor(
        log_file="logs/training_gpu_stats/threshold_monitor.log",
        interval=5,
        memory_threshold=80.0,
        display=True,
        verbose=False
    )
    
    monitor.start()
    print(f"Memory threshold monitor started (threshold: 80%)")
    print("Running for 30 seconds...")
    
    try:
        time.sleep(30)
    finally:
        monitor.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPU Memory Monitor Integration Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_monitor_integration.py --mode training
  python gpu_monitor_integration.py --mode multiprocess
  python gpu_monitor_integration.py --mode callback
        """
    )
    parser.add_argument(
        "--mode",
        choices=["training", "multiprocess", "callback"],
        default="training",
        help="Which integration example to run"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs/training_gpu_stats", exist_ok=True)
    
    if args.mode == "training":
        example_training_integration()
    elif args.mode == "multiprocess":
        example_multi_process_training()
    elif args.mode == "callback":
        example_monitor_with_callbacks()
