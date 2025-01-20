import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from analysis.profiler import PerformanceProfiler, profile_memory_usage

@pytest.fixture
def profiler():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PerformanceProfiler(log_dir=tmpdir)

@pytest.fixture
def dummy_model():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
        
        def forward(self, x):
            time.sleep(0.01)  # Simulate computation
            return self.linear(x)
    
    return DummyModel()

def test_profiler_initialization(profiler):
    """Test profiler initialization"""
    assert isinstance(profiler.metrics['inference_times'], list)
    assert isinstance(profiler.metrics['memory_usage'], list)
    assert Path(profiler.log_dir).exists()

def test_model_profiling(profiler, dummy_model):
    """Test model profiling functionality"""
    input_data = (torch.randn(32, 10),)
    metrics = profiler.profile_torch_model(dummy_model, input_data)
    
    assert 'avg_inference_time' in metrics
    assert 'max_memory_usage' in metrics
    assert metrics['avg_inference_time'] > 0
    assert metrics['max_memory_usage'] >= 0

def test_resource_monitoring(profiler):
    """Test resource monitoring"""
    metrics = profiler.monitor_resources()
    
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 0 <= metrics['cpu_percent'] <= 100
    assert 0 <= metrics['memory_percent'] <= 100

@profile_memory_usage
def memory_intensive_function():
    """Function to test memory profiling"""
    large_array = np.random.randn(1000, 1000)
    time.sleep(0.1)
    return large_array.shape

def test_memory_profiling():
    """Test memory profiling decorator"""
    result = memory_intensive_function()
    assert result == (1000, 1000)

def test_performance_summary(profiler):
    """Test performance summary generation"""
    # Add some dummy data
    profiler.metrics['inference_times'] = [0.1, 0.2, 0.3]
    profiler.metrics['cpu_usage'] = [20, 30, 40]
    profiler.metrics['memory_usage'] = [50, 60, 70]
    
    summary = profiler.get_performance_summary()
    
    assert 'inference_time' in summary
    assert 'resource_usage' in summary
    assert abs(summary['inference_time']['mean'] - 0.2) < 1e-6
    assert abs(summary['resource_usage']['cpu_mean'] - 30) < 1e-6

def test_batch_processing_stats(profiler):
    """Test batch processing statistics"""
    profiler.metrics['batch_processing_times'] = [
        {'batch_size': 32, 'time_ms': 10},
        {'batch_size': 64, 'time_ms': 20}
    ]
    
    stats = profiler._get_batch_stats()
    
    assert 'avg_batch_time' in stats
    assert 'throughput' in stats
    assert stats['total_batches'] == 2
    assert stats['avg_batch_size'] == 48

def test_continuous_monitoring(profiler):
    """Test continuous monitoring (short duration)"""
    def mock_monitoring():
        profiler.start_continuous_monitoring(interval=1)
    
    # Start monitoring in a separate thread
    import threading
    monitor_thread = threading.Thread(target=mock_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Let it run briefly
    time.sleep(2)
    
    # Check that metrics were collected
    assert len(profiler.metrics['cpu_usage']) > 0
    assert len(profiler.metrics['memory_usage']) > 0

def test_profile_function_decorator(profiler):
    """Test function profiling decorator"""
    @profiler.profile_function
    def test_function():
        time.sleep(0.1)
        return 42
    
    result = test_function()
    assert result == 42
    
    # Check that profiling files were created
    log_files = list(Path(profiler.log_dir).glob("test_function_*.stats"))
    assert len(log_files) > 0

if __name__ == '__main__':
    pytest.main([__file__])