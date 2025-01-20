import cProfile
import pstats
from functools import wraps
import time
import torch
import torch.autograd.profiler as torch_profiler
from typing import Callable, Any, Dict, List
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import psutil
import logging
from rich.console import Console
from rich.table import Table

class PerformanceProfiler:
    def __init__(self, log_dir: str = "logs/performance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'gpu_usage': [],
            'cpu_usage': [],
            'batch_processing_times': [],
        }
        
        logging.basicConfig(
            filename=self.log_dir / 'performance.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def profile_function(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            try:
                result = profiler.runcall(func, *args, **kwargs)
                stats = pstats.Stats(profiler)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                stats_file = self.log_dir / f"{func.__name__}_{timestamp}.stats"
                stats.dump_stats(str(stats_file))
                
                self.logger.info(f"Function {func.__name__} profiling saved to {stats_file}")
                return result
            except Exception as e:
                self.logger.error(f"Error profiling {func.__name__}: {e}")
                raise
        return wrapper
    
    def profile_torch_model(self, model: torch.nn.Module, input_data: tuple) -> Dict:
        with torch_profiler.profile(
            activities=[
                torch_profiler.ProfilerActivity.CPU,
                torch_profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            with_stack=True
        ) as prof:
            # Warmup
            for _ in range(3):
                model(*input_data)
            
            # Profile runs
            times = []
            memory_usage = []
            
            for _ in range(10):
                start = time.perf_counter()
                model(*input_data)
                times.append(time.perf_counter() - start)
                memory_usage.append(torch.cuda.max_memory_allocated())
        
        metrics = {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'max_memory_usage': max(memory_usage),
            'trace_summary': str(prof.key_averages().table()),
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prof.export_chrome_trace(str(self.log_dir / f"trace_{timestamp}.json"))
        
        return metrics
    
    def monitor_resources(self) -> Dict:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used,
            'memory_available': memory.available
        }
        
        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_used': torch.cuda.memory_allocated(),
                'gpu_memory_cached': torch.cuda.memory_reserved()
            })
        
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage'].append(memory.percent)
        
        return metrics
    
    def get_performance_summary(self) -> Dict:
        inference_times = np.array(self.metrics['inference_times'])
        cpu_usage = np.array(self.metrics['cpu_usage'])
        memory_usage = np.array(self.metrics['memory_usage'])
        
        summary = {
            'inference_time': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times),
                'p95': np.percentile(inference_times, 95)
            },
            'resource_usage': {
                'cpu_mean': np.mean(cpu_usage),
                'cpu_max': np.max(cpu_usage),
                'memory_mean': np.mean(memory_usage),
                'memory_max': np.max(memory_usage)
            },
            'batch_processing': self._get_batch_stats()
        }
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(self.log_dir / f"summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _get_batch_stats(self) -> Dict:
        if not self.metrics['batch_processing_times']:
            return {}
        
        batch_times = [x['time_ms'] for x in self.metrics['batch_processing_times']]
        batch_sizes = [x['batch_size'] for x in self.metrics['batch_processing_times']]
        
        return {
            'avg_batch_time': np.mean(batch_times),
            'max_batch_time': np.max(batch_times),
            'avg_batch_size': np.mean(batch_sizes),
            'total_batches': len(batch_times),
            'throughput': np.sum(batch_sizes) / np.sum(batch_times)
        }
    
    def display_live_metrics(self):
        """Display live performance metrics in the terminal"""
        table = Table(title="Performance Metrics")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        summary = self.get_performance_summary()
        
        # Inference metrics
        table.add_row(
            "Avg Inference Time (ms)", 
            f"{summary['inference_time']['mean']:.2f}"
        )
        table.add_row(
            "95th Percentile (ms)", 
            f"{summary['inference_time']['p95']:.2f}"
        )
        
        # Resource usage
        table.add_row(
            "CPU Usage (%)", 
            f"{summary['resource_usage']['cpu_mean']:.1f}"
        )
        table.add_row(
            "Memory Usage (%)", 
            f"{summary['resource_usage']['memory_mean']:.1f}"
        )
        
        # Batch processing
        if 'batch_processing' in summary:
            table.add_row(
                "Throughput (samples/ms)",
                f"{summary['batch_processing'].get('throughput', 0):.2f}"
            )
        
        self.console.clear()
        self.console.print(table)
    
    def start_continuous_monitoring(self, interval: int = 5):
        """Start continuous monitoring with specified interval"""
        try:
            while True:
                metrics = self.monitor_resources()
                self.display_live_metrics()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")

def profile_memory_usage(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss
            memory_diff = memory_after - memory_before
            
            logging.info(
                f"Memory usage for {func.__name__}: "
                f"Before: {memory_before/1024/1024:.2f}MB, "
                f"After: {memory_after/1024/1024:.2f}MB, "
                f"Difference: {memory_diff/1024/1024:.2f}MB"
            )
            
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            raise
            
    return wrapper