# src/agents/multi_agent_coordinator.py
import numpy as np
from typing import Dict, List, Tuple
import torch
import logging
from dataclasses import dataclass
from .deep_q_agent import DeepQAgent, DeepQAgentConfig

@dataclass
class MarketRegime:
    type: str
    confidence: float
    indicators: Dict[str, float]

class MultiAgentCoordinator:
    """
    Coordinates multiple specialized agents for different market conditions
    """
    def __init__(self, base_config: DeepQAgentConfig, 
                 specialist_configs: Dict[str, Dict],
                 voting_weights: Dict[str, float] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize base agent
        self.base_agent = DeepQAgent(base_config)
        
        # Initialize specialist agents
        self.specialist_agents = {}
        for specialist_type, config_updates in specialist_configs.items():
            specialist_config = base_config.__class__(**vars(base_config))
            # Update config with specialist-specific parameters
            for key, value in config_updates.items():
                setattr(specialist_config, key, value)
            self.specialist_agents[specialist_type] = DeepQAgent(specialist_config)
            
        # Set voting weights
        if voting_weights is None:
            num_agents = 1 + len(self.specialist_agents)
            self.voting_weights = {
                'base': 1.0 / num_agents,
                **{name: 1.0 / num_agents for name in specialist_configs.keys()}
            }
        else:
            self.voting_weights = voting_weights
            
        self.market_regime = None
        self.confidence_threshold = 0.6
        self.performance_history = {
            'base': [],
            **{name: [] for name in specialist_configs.keys()}
        }
        
    def detect_market_regime(self, state: np.ndarray) -> MarketRegime:
        """
        Detect current market regime based on state
        Returns:
            MarketRegime object containing regime type and confidence
        """
        # Calculate key indicators
        volatility = np.std(state[-20:])
        trend = np.mean(state[-20:] - state[-21:-1])
        momentum = np.sum(np.diff(state[-10:]) > 0) / 9
        volume_trend = np.mean(np.diff(state[-20:]))
        
        # Calculate regime probabilities
        regime_scores = {
            'volatile': self._calculate_volatility_score(volatility, volume_trend),
            'trending': self._calculate_trend_score(trend, momentum),
            'ranging': self._calculate_range_score(volatility, trend),
            'breakout': self._calculate_breakout_score(state)
        }
        
        # Select regime with highest probability
        regime_type = max(regime_scores.items(), key=lambda x: x[1])[0]
        confidence = regime_scores[regime_type]
        
        return MarketRegime(
            type=regime_type,
            confidence=confidence,
            indicators={
                'volatility': volatility,
                'trend': trend,
                'momentum': momentum,
                'volume_trend': volume_trend
            }
        )
        
    def _calculate_volatility_score(self, volatility: float, volume_trend: float) -> float:
        """Calculate probability of volatile regime"""
        vol_score = np.clip(volatility / 0.02, 0, 1)
        volume_score = np.clip(abs(volume_trend) / 0.01, 0, 1)
        return 0.7 * vol_score + 0.3 * volume_score
        
    def _calculate_trend_score(self, trend: float, momentum: float) -> float:
        """Calculate probability of trending regime"""
        trend_score = np.clip(abs(trend) / 0.001, 0, 1)
        momentum_score = np.clip(abs(momentum - 0.5) * 2, 0, 1)
        return 0.6 * trend_score + 0.4 * momentum_score
        
    def _calculate_range_score(self, volatility: float, trend: float) -> float:
        """Calculate probability of ranging regime"""
        vol_score = 1 - np.clip(volatility / 0.02, 0, 1)
        trend_score = 1 - np.clip(abs(trend) / 0.001, 0, 1)
        return 0.5 * vol_score + 0.5 * trend_score
        
    def _calculate_breakout_score(self, state: np.ndarray) -> float:
        """Calculate probability of breakout regime"""
        # Calculate Bollinger Bands
        rolling_mean = np.mean(state[-20:])
        rolling_std = np.std(state[-20:])
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        
        # Check for breakout conditions
        current_price = state[-1]
        price_position = abs(current_price - rolling_mean) / rolling_std
        volume_surge = np.mean(state[-5:]) / np.mean(state[-20:])
        
        return np.clip(0.6 * price_position + 0.4 * volume_surge, 0, 1)
        
    def select_action(self, state: np.ndarray, 
                     training: bool = True) -> Tuple[int, Dict[str, float]]:
        """
        Select action using ensemble of agents
        Returns:
            Tuple of (selected action, action probabilities)
        """
        # Detect market regime
        self.market_regime = self.detect_market_regime(state)
        
        # Get Q-values from all agents
        base_q_values = self._get_q_values(self.base_agent, state)
        specialist_q_values = {
            name: self._get_q_values(agent, state)
            for name, agent in self.specialist_agents.items()
        }
        
        # Adjust weights based on regime and performance
        adjusted_weights = self._adjust_weights_for_regime(
            self.voting_weights.copy(),
            self.market_regime
        )
        
        # Combine Q-values using weighted voting
        combined_q_values = np.zeros_like(base_q_values)
        combined_q_values += adjusted_weights['base'] * base_q_values
        
        for name, q_values in specialist_q_values.items():
            combined_q_values += adjusted_weights[name] * q_values
            
        # Select action
        if training and np.random.random() < self.base_agent.epsilon:
            action = np.random.randint(0, len(combined_q_values))
        else:
            action = np.argmax(combined_q_values)
            
        # Calculate action probabilities using softmax
        action_probs = self._softmax(combined_q_values)
        
        return action, {
            'action_probs': action_probs,
            'regime': self.market_regime.type,
            'confidence': self.market_regime.confidence,
            'weights': adjusted_weights
        }
        
    def _get_q_values(self, agent: DeepQAgent, state: np.ndarray) -> np.ndarray:
        """Get Q-values from agent"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.config.device)
            return agent.policy_net(state_tensor).cpu().numpy()[0]
            
    def _adjust_weights_for_regime(self, weights: Dict[str, float], 
                                 regime: MarketRegime) -> Dict[str, float]:
        """Adjust voting weights based on market regime and agent performance"""
        # Boost weights for specialists in their preferred regime
        regime_specialists = {
            'volatile': ['volatile_specialist'],
            'trending': ['trend_follower'],
            'ranging': ['range_trader'],
            'breakout': ['breakout_trader']
        }
        
        # Get relevant specialists for current regime
        relevant_specialists = regime_specialists.get(regime.type, [])
        
        # Boost weights for relevant specialists
        boost_factor = 1 + regime.confidence
        for specialist in relevant_specialists:
            if specialist in weights:
                weights[specialist] *= boost_factor
                
        # Normalize weights
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}
        
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update all agents"""
        # Update base agent
        self.base_agent.update_memory(state, action, reward, next_state, done)
        self.base_agent.train_step()
        
        # Update specialist agents
        for agent in self.specialist_agents.values():
            agent.update_memory(state, action, reward, next_state, done)
            agent.train_step()
            
        # Update performance history
        if done:
            self._update_performance_history(reward)
            
    def _update_performance_history(self, episode_reward: float):
        """Update performance history for all agents"""
        self.performance_history['base'].append(episode_reward)
        for name in self.specialist_agents.keys():
            self.performance_history[name].append(episode_reward)
            
        # Keep only recent history
        max_history = 100
        for history in self.performance_history.values():
            if len(history) > max_history:
                history.pop(0)
                
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each set of scores"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    def save_agents(self, path: str):
        """Save all agents"""
        self.base_agent.save(f"{path}/base_agent.pth")
        for name, agent in self.specialist_agents.items():
            agent.save(f"{path}/{name}.pth")
            
    def load_agents(self, path: str):
        """Load all agents"""
        self.base_agent.load(f"{path}/base_agent.pth")
        for name, agent in self.specialist_agents.items():
            agent.load(f"{path}/{name}.pth")