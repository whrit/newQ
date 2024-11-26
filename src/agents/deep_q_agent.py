# src/agents/deep_q_agent.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from ..models.deep_q_network import DuelingDQN, NetworkConfig
from .experience_buffer import PrioritizedReplayBuffer
from config.trading_config import ConfigManager

@dataclass
class DeepQAgentConfig:
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    learning_rate: float
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_size: int = 100000
    update_frequency: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DeepQAgent:
    def __init__(self, state_dim: int, action_dim: int, custom_config: Dict = None):
        # Load configuration
        config_manager = ConfigManager()
        base_config = config_manager.get_deep_q_config()
        
        # Merge with custom config if provided
        self.config = DeepQAgentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            **{**base_config, **(custom_config or {})}
        )
        
        # Rest of initialization remains the same
        self.logger = logging.getLogger(__name__)
        
        network_config = NetworkConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            learning_rate=self.config.learning_rate,
            device=self.config.device
        )
        
        self.policy_net = DuelingDQN(network_config).to(config.device)
        self.target_net = DuelingDQN(network_config).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(
            config.buffer_size,
            config.batch_size,
            config.device
        )
        
        self.epsilon = config.initial_epsilon
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