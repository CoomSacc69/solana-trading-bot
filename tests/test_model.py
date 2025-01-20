import pytest
import numpy as np
import torch
from pathlib import Path
import sys
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.model import HybridModel, ModelConfig, TradingModelManager

@pytest.fixture
def model_config():
    return ModelConfig(
        lstm_hidden_size=64,
        transformer_hidden_size=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        sequence_length=30,
        num_features=10
    )

@pytest.fixture
def model(model_config):
    return HybridModel(model_config)

@pytest.fixture
def model_manager(model_config):
    return TradingModelManager(model_config)

@pytest.fixture
def sample_data():
    # Generate sample data
    batch_size = 16
    seq_length = 30
    num_features = 10
    
    sequential_data = np.random.randn(batch_size, seq_length, num_features)
    features = np.random.randn(batch_size, num_features)
    targets = np.random.randint(0, 3, size=batch_size)
    vol_targets = np.random.randn(batch_size, 1)
    price_targets = np.random.randn(batch_size, 1)
    
    return sequential_data, features, targets, vol_targets, price_targets

def test_model_initialization(model):
    """Test model initialization"""
    assert isinstance(model, HybridModel)
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert isinstance(model.transformer, torch.nn.TransformerEncoder)

def test_model_forward(model, sample_data):
    """Test model forward pass"""
    sequential_data, features, _, _, _ = sample_data
    
    # Convert to tensors
    seq_tensor = torch.FloatTensor(sequential_data)
    features_tensor = torch.FloatTensor(features)
    
    # Forward pass
    main_output, aux_outputs = model(seq_tensor, features_tensor)
    
    # Check outputs
    assert main_output.shape == (16, 3)  # Batch size x num_classes
    assert 'volatility' in aux_outputs
    assert 'price_change' in aux_outputs
    assert aux_outputs['volatility'].shape == (16, 1)
    assert aux_outputs['price_change'].shape == (16, 1)

def test_model_training_step(model_manager, sample_data):
    """Test training step"""
    losses = model_manager.train_step(sample_data)
    
    assert 'total_loss' in losses
    assert 'main_loss' in losses
    assert 'volatility_loss' in losses
    assert 'price_loss' in losses
    assert all(isinstance(v, float) for v in losses.values())
    assert all(v >= 0 for v in losses.values())

def test_model_prediction(model_manager, sample_data):
    """Test model prediction"""
    sequential_data, features, _, _, _ = sample_data
    
    predictions, aux_predictions = model_manager.predict(sequential_data, features)
    
    assert predictions.shape == (16, 3)  # Batch size x num_classes
    assert np.allclose(predictions.sum(axis=1), 1.0)  # Check probabilities sum to 1
    assert all(0 <= p <= 1 for p in predictions.flatten())
    
    assert 'volatility' in aux_predictions
    assert 'price_change' in aux_predictions

def test_model_save_load(model_manager, sample_data):
    """Test model saving and loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        save_path = Path(tmpdir) / "test_model.pt"
        model_manager.save_model(str(save_path))
        
        # Make predictions before loading
        orig_predictions, _ = model_manager.predict(sample_data[0], sample_data[1])
        
        # Create new manager and load model
        new_manager = TradingModelManager(model_manager.config)
        new_manager.load_model(str(save_path))
        
        # Make predictions with loaded model
        loaded_predictions, _ = new_manager.predict(sample_data[0], sample_data[1])
        
        # Check predictions match
        assert np.allclose(orig_predictions, loaded_predictions)

def test_model_gpu_support(model_config):
    """Test GPU support if available"""
    if torch.cuda.is_available():
        model = HybridModel(model_config).cuda()
        assert next(model.parameters()).is_cuda
        
        # Test forward pass on GPU
        batch_size = 16
        seq_length = 30
        num_features = 10
        
        seq_tensor = torch.randn(batch_size, seq_length, num_features).cuda()
        features_tensor = torch.randn(batch_size, num_features).cuda()
        
        main_output, aux_outputs = model(seq_tensor, features_tensor)
        assert main_output.is_cuda
        assert all(v.is_cuda for v in aux_outputs.values())

def test_model_gradient_flow(model_manager):
    """Test gradient flow through the model"""
    # Generate small batch
    batch_size = 4
    seq_length = 30
    num_features = 10
    
    sequential_data = np.random.randn(batch_size, seq_length, num_features)
    features = np.random.randn(batch_size, num_features)
    targets = np.random.randint(0, 3, size=batch_size)
    vol_targets = np.random.randn(batch_size, 1)
    price_targets = np.random.randn(batch_size, 1)
    
    # Initial parameters
    initial_params = [p.clone().detach() for p in model_manager.model.parameters()]
    
    # Training step
    model_manager.train_step((sequential_data, features, targets, vol_targets, price_targets))
    
    # Check parameters have been updated
    current_params = [p.clone().detach() for p in model_manager.model.parameters()]
    
    # Verify at least some parameters changed
    assert any(not torch.equal(i, c) for i, c in zip(initial_params, current_params))

def test_attention_mechanism(model):
    """Test attention mechanism"""
    batch_size = 4
    seq_length = 30
    num_features = 10
    
    # Create input with a clear pattern
    sequential_data = np.zeros((batch_size, seq_length, num_features))
    sequential_data[:, -5:, :] = 1.0  # Make last 5 timesteps significant
    
    features = np.random.randn(batch_size, num_features)
    
    # Convert to tensors
    seq_tensor = torch.FloatTensor(sequential_data)
    features_tensor = torch.FloatTensor(features)
    
    # Get attention weights (would need to modify model to return attention weights)
    main_output, _ = model(seq_tensor, features_tensor)
    
    # Basic shape test
    assert main_output.shape == (batch_size, 3)

def test_model_robustness(model_manager):
    """Test model robustness to various input scenarios"""
    batch_size = 4
    seq_length = 30
    num_features = 10
    
    # Test with zeros
    zeros_data = np.zeros((batch_size, seq_length, num_features))
    zeros_features = np.zeros((batch_size, num_features))
    predictions, _ = model_manager.predict(zeros_data, zeros_features)
    assert not np.any(np.isnan(predictions))
    
    # Test with very large values
    large_data = np.random.randn(batch_size, seq_length, num_features) * 1000
    large_features = np.random.randn(batch_size, num_features) * 1000
    predictions, _ = model_manager.predict(large_data, large_features)
    assert not np.any(np.isnan(predictions))
    
    # Test with missing values (NaN)
    nan_data = np.random.randn(batch_size, seq_length, num_features)
    nan_data[0, 0, 0] = np.nan  # Add a NaN
    nan_features = np.random.randn(batch_size, num_features)
    
    with pytest.raises(ValueError):
        model_manager.predict(nan_data, nan_features)

if __name__ == '__main__':
    pytest.main([__file__])