import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
from loguru import logger
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for the hybrid model"""
    lstm_hidden_size: int = 128
    transformer_hidden_size: int = 256
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 50
    num_features: int = 15

class AttentionLayer(nn.Module):
    """Custom attention layer for time series"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_output)

class HybridModel(nn.Module):
    """Hybrid model combining LSTM, Transformer, and traditional features"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # LSTM for sequential price data
        self.lstm = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=2,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer for pattern recognition
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.transformer_hidden_size)
        )
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.transformer_hidden_size * 4,
            dropout=config.dropout_rate
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Custom attention layer
        self.attention = AttentionLayer(config.transformer_hidden_size)
        
        # Feature processing
        self.feature_embedding = nn.Linear(
            config.num_features, 
            config.transformer_hidden_size
        )
        
        # Combine all features
        combined_size = (
            config.lstm_hidden_size * 2 +  # Bidirectional LSTM
            config.transformer_hidden_size +
            config.num_features  # Original features
        )
        
        # Final prediction layers
        self.prediction_layers = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 3)  # 3 classes: buy, sell, hold
        )
        
        # Auxiliary tasks
        self.volatility_predictor = nn.Linear(combined_size, 1)
        self.price_predictor = nn.Linear(combined_size, 1)
        
    def forward(
        self, 
        x_seq: torch.Tensor,
        x_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        Args:
            x_seq: Sequential data (batch_size, seq_len, num_features)
            x_features: Additional features (batch_size, num_features)
        Returns:
            Main prediction and auxiliary outputs
        """
        # Process sequential data with LSTM
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Process with Transformer
        x_transformed = self.feature_embedding(x_seq)
        x_transformed = x_transformed + self.position_embedding
        transformer_out = self.transformer(x_transformed)
        transformer_out = self.attention(transformer_out)
        transformer_out = transformer_out.mean(dim=1)  # Pool across time
        
        # Combine all features
        combined = torch.cat([lstm_out, transformer_out, x_features], dim=1)
        
        # Main prediction
        main_output = self.prediction_layers(combined)
        
        # Auxiliary predictions
        aux_outputs = {
            'volatility': self.volatility_predictor(combined),
            'price_change': self.price_predictor(combined)
        }
        
        return main_output, aux_outputs

class TradingModelManager:
    """Manages model training, evaluation, and inference"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HybridModel(config).to(self.device)
        self.scaler = StandardScaler()
        self.model_path = Path('models')
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=100,
            steps_per_epoch=100
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        logger.info(f"Model initialized on device: {self.device}")
        
    def preprocess_data(
        self,
        sequential_data: np.ndarray,
        features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data for model input"""
        # Scale sequential data
        seq_shape = sequential_data.shape
        seq_flat = sequential_data.reshape(-1, seq_shape[-1])
        seq_scaled = self.scaler.fit_transform(seq_flat)
        seq_scaled = seq_scaled.reshape(seq_shape)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensors
        seq_tensor = torch.FloatTensor(seq_scaled).to(self.device)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        return seq_tensor, features_tensor
    
    def train_step(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Unpack batch
        seq_data, features, targets, vol_targets, price_targets = batch
        
        # Preprocess
        seq_tensor, features_tensor = self.preprocess_data(seq_data, features)
        targets_tensor = torch.LongTensor(targets).to(self.device)
        vol_tensor = torch.FloatTensor(vol_targets).to(self.device)
        price_tensor = torch.FloatTensor(price_targets).to(self.device)
        
        # Forward pass
        main_output, aux_outputs = self.model(seq_tensor, features_tensor)
        
        # Calculate losses
        main_loss = self.classification_loss(main_output, targets_tensor)
        vol_loss = self.regression_loss(aux_outputs['volatility'], vol_tensor)
        price_loss = self.regression_loss(aux_outputs['price_change'], price_tensor)
        
        # Combined loss
        total_loss = main_loss + 0.2 * vol_loss + 0.2 * price_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'volatility_loss': vol_loss.item(),
            'price_loss': price_loss.item()
        }
    
    def predict(
        self,
        sequential_data: np.ndarray,
        features: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions"""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess
            seq_tensor, features_tensor = self.preprocess_data(
                sequential_data, features
            )
            
            # Get predictions
            main_output, aux_outputs = self.model(seq_tensor, features_tensor)
            
            # Convert to numpy
            predictions = torch.softmax(main_output, dim=1).cpu().numpy()
            aux_predictions = {
                k: v.cpu().numpy() for k, v in aux_outputs.items()
            }
            
        return predictions, aux_predictions
    
    def save_model(self, path: str):
        """Save model and preprocessing objects"""
        save_path = self.model_path / path
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, save_path)
        
        # Save scaler
        joblib.dump(self.scaler, self.model_path / f'{path}_scaler.pkl')
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load model and preprocessing objects"""
        load_path = self.model_path / path
        
        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")
            
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        self.scaler = joblib.load(self.model_path / f'{path}_scaler.pkl')
        logger.info(f"Model loaded from {load_path}")

# Additional utilities for model evaluation
def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """Calculate model performance metrics"""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score
    )
    
    # Get class predictions
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, pred_classes, average='weighted'
    )
    
    # Calculate ROC AUC for each class
    auc_scores = []
    for i in range(predictions.shape[1]):
        try:
            auc = roc_auc_score(targets == i, predictions[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_scores': np.mean(auc_scores)
    }