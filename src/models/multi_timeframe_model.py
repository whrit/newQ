# src/models/multi_timeframe_model.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer

@dataclass
class TimeframeConfig:
    timeframes: List[str]  # e.g., ["1m", "5m", "15m", "1h", "4h", "1d"]
    feature_dims: Dict[str, int]
    hidden_dims: List[int]
    output_dim: int
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 0.001

class MultiTimeframeAttention(nn.Module):
    """
    Attention mechanism for combining multiple timeframe features
    """
    def __init__(self, input_dim: int, attention_dim: int):
        super(MultiTimeframeAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism
        Args:
            x: Input tensor of shape (batch_size, num_timeframes, feature_dim)
        Returns:
            Weighted sum of features and attention weights
        """
        attention_weights = self.attention(x)
        weighted_features = torch.sum(x * attention_weights, dim=1)
        return weighted_features, attention_weights

class TimeframeEncoder(nn.Module):
    """
    Encoder for individual timeframe data
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TimeframeEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class MultiTimeframeModel(nn.Module):
    """
    Complete multi-timeframe model combining attention and transformer mechanisms
    """
    def __init__(self, config: TimeframeConfig):
        super(MultiTimeframeModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create encoders for each timeframe
        self.timeframe_encoders = nn.ModuleDict({
            tf: TimeframeEncoder(config.feature_dims[tf], config.hidden_dims[0])
            for tf in config.timeframes
        })
        
        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dims[0],
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dims[1],
            dropout=config.dropout
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        
        # Attention mechanism
        self.attention = MultiTimeframeAttention(
            config.hidden_dims[0], 
            config.hidden_dims[1]
        )
        
        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], config.output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
                
    def forward(self, timeframe_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        Args:
            timeframe_data: Dictionary mapping timeframes to feature tensors
        Returns:
            Dictionary containing predictions and attention weights
        """
        try:
            # Encode each timeframe's data
            encoded_features = []
            for timeframe in self.config.timeframes:
                if timeframe not in timeframe_data:
                    raise KeyError(f"Missing data for timeframe {timeframe}")
                    
                features = timeframe_data[timeframe]
                encoded = self.timeframe_encoders[timeframe](features)
                encoded_features.append(encoded)
                
            # Stack encoded features
            stacked_features = torch.stack(encoded_features, dim=1)
            
            # Apply transformer
            transformer_out = self.transformer(stacked_features)
            
            # Apply attention mechanism
            attended_features, attention_weights = self.attention(transformer_out)
            
            # Generate predictions
            predictions = self.prediction_head(attended_features)
            
            return {
                'predictions': predictions,
                'attention_weights': attention_weights,
                'transformed_features': transformer_out
            }
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise
            
    def calculate_time_importance(self, timeframe_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate the importance of each timeframe based on attention weights
        """
        with torch.no_grad():
            outputs = self.forward(timeframe_data)
            attention_weights = outputs['attention_weights'].squeeze().cpu().numpy()
            
            return {
                timeframe: float(weight)
                for timeframe, weight in zip(self.config.timeframes, attention_weights)
            }
            
class MultiTimeframeModelTrainer:
    """
    Trainer class for MultiTimeframeModel
    """
    def __init__(self, model: MultiTimeframeModel, config: TimeframeConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.torch.backends.mps.is_available else "cpu")
        self.model.to(self.device)
        
    def train_step(self, batch: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(batch)
        predictions = outputs['predictions']
        
        # Calculate losses
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # Add regularization if needed
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        
        total_loss = mse_loss + 0.01 * l2_reg
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'l2_reg': l2_reg.item()
        }
        
    def validate(self, val_data: Dict[str, torch.Tensor], val_targets: torch.Tensor) -> Dict[str, float]:
        """
        Validation step
        """
        self.model.eval()
        with torch.no_grad():
            val_data = {k: v.to(self.device) for k, v in val_data.items()}
            val_targets = val_targets.to(self.device)
            
            outputs = self.model(val_data)
            predictions = outputs['predictions']
            
            val_loss = nn.MSELoss()(predictions, val_targets)
            
            # Calculate additional metrics
            mae = nn.L1Loss()(predictions, val_targets)
            
            return {
                'val_loss': val_loss.item(),
                'val_mae': mae.item()
            }
            
    def save_checkpoint(self, path: str, epoch: int, best_loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint