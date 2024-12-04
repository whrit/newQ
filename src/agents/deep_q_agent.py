# src/agents/deep_q_agent.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
import logging
from ..models.deep_q_network import DuelingDQN, NetworkConfig
from .experience_buffer import PrioritizedReplayBuffer
from config.trading_config import ConfigManager

@dataclass
class DeepQAgentConfig:
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])  # Better for complex market data
    learning_rate: float = 0.0001  # Reduced for more stable learning
    gamma: float = 0.99  # Future reward discount
    tau: float = 0.001  # Slower target updates for stability
    initial_epsilon: float = 0.9  # High initial exploration
    final_epsilon: float = 0.05  # Higher final exploration for market changes
    epsilon_decay: float = 0.999  # Slower decay
    batch_size: int = 128  # Larger batch size for better stability
    buffer_size: int = 500000  # Larger buffer for more experience
    update_frequency: int = 10  # More frequent updates
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.torch.backends.mps.is_available else "cpu"
    warmup_steps: int = 10000  # Steps before training starts
    entropy_weight: float = 0.01  # For exploration bonus
    target_update_interval: int = 1000  # Hard update interval

class DeepQAgent:
    def __init__(
        self, 
        config_or_state_dim: Union[DeepQAgentConfig, int],
        action_dim: int = None,
        custom_config: Dict = None
    ):
        """
        Initialize DeepQAgent with either a config object or individual parameters
        
        Args:
            config_or_state_dim: Either a DeepQAgentConfig object or state dimension
            action_dim: Action dimension (required if first arg is state_dim)
            custom_config: Optional custom configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Handle initialization with config object
        if isinstance(config_or_state_dim, DeepQAgentConfig):
            self.config = config_or_state_dim
        # Handle initialization with separate parameters
        else:
            if action_dim is None:
                raise ValueError("action_dim must be provided when initializing with state_dim")
                
            config_manager = ConfigManager()
            base_config = config_manager.get_deep_q_config()
            
            self.config = DeepQAgentConfig(
                state_dim=config_or_state_dim,
                action_dim=action_dim,
                **{**base_config, **(custom_config or {})}
            )
        
        # Initialize networks
        network_config = NetworkConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            learning_rate=self.config.learning_rate,
            device=self.config.device
        )
        
        self.policy_net = DuelingDQN(network_config).to(self.config.device)
        self.target_net = DuelingDQN(network_config).to(self.config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.config.batch_size,
            self.config.device
        )
        
        self.epsilon = self.config.initial_epsilon
        self.steps = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy with noisy networks
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_dim)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
            
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step
        """
        if len(self.memory) < self.config.batch_size:
            return {}
            
        # Sample batch from replay buffer
        batch, weights, indices = self.memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).to(self.config.device)
        weights = torch.FloatTensor(weights).to(self.config.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using double Q-learning
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + \
                            (1 - dones.unsqueeze(1)) * self.config.gamma * next_q_values
                            
        # Compute loss with importance sampling weights
        td_errors = torch.abs(current_q_values - target_q_values).detach()
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.squeeze().cpu().numpy())
        
        # Soft update target network
        if self.steps % self.config.update_frequency == 0:
            self._soft_update_target_network()
            
        # Decay epsilon
        self.epsilon = max(
            self.config.final_epsilon,
            self.epsilon * self.config.epsilon_decay
        )
        
        self.steps += 1
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'mean_q': current_q_values.mean().item()
        }
        
    def _soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), 
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * policy_param.data + 
                (1 - self.config.tau) * target_param.data
            )
            
    def save(self, path: str):
        """Save agent's state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """Load agent's state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
    def update_memory(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.memory.add(state, action, reward, next_state, done)