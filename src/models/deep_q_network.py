# src/models/deep_q_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    learning_rate: float
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.torch.backends.mps.is_available else "cpu"

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with noisy linear layers for better exploration
    """
    def __init__(self, config: NetworkConfig):
        super(DuelingDQN, self).__init__()
        self.config = config
        
        # Feature layers
        self.features = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dims[0]),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dims[1]),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            NoisyLinear(config.hidden_dims[2], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            NoisyLinear(config.hidden_dims[2], config.action_dim)
        )
        
        self.to(config.device)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        Args:
            state: Current state tensor
        Returns:
            Q-values for each action
        """
        features = self.features(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using dueling architecture
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        
    def reset_noise(self):
        """Reset noise for exploration"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Scale noise for better exploration"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

class DQNEnsemble:
    """
    Ensemble of DQN models for better stability and uncertainty estimation
    """
    def __init__(self, config: NetworkConfig, num_models: int = 3):
        self.models = [DuelingDQN(config) for _ in range(num_models)]
        self.optimizers = [torch.optim.Adam(model.parameters(), 
                                          lr=config.learning_rate) for model in self.models]
        self.config = config
        
    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Get ensemble prediction and uncertainty
        Args:
            state: Current state tensor
        Returns:
            mean Q-values and uncertainty measure
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions.append(model(state))
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0).mean().item()
        
        return mean_prediction, uncertainty
    
    def train_step(self, batch: Tuple) -> float:
        """
        Train all models in ensemble
        Args:
            batch: Tuple of (state, action, reward, next_state, done)
        Returns:
            Mean loss across ensemble
        """
        losses = []
        states, actions, rewards, next_states, dones = batch
        
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            
            # Current Q-values
            current_q = model(states).gather(1, actions)
            
            # Next Q-values with double Q-learning
            with torch.no_grad():
                # Get actions from current model
                next_actions = model(next_states).argmax(dim=1, keepdim=True)
                # Get Q-values from target model
                next_q = model(next_states).gather(1, next_actions)
                target_q = rewards + (1 - dones) * 0.99 * next_q
            
            # Compute loss and update
            loss = F.smooth_l1_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Reset noise for exploration
            model.reset_noise()
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def save(self, path: str):
        """Save ensemble models"""
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{path}/model_{i}.pth")
    
    def load(self, path: str):
        """Load ensemble models"""
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(f"{path}/model_{i}.pth"))