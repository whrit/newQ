# src/models/agent_manager.py

import os
import logging
from typing import Optional, List
from datetime import datetime
from ..agents.deep_q_agent import DeepQAgent, DeepQAgentConfig
import numpy as np
import pandas as pd

class AgentManager:
    def __init__(self, save_dir: str = "models/saved_agents"):
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.agent_path = os.path.join(save_dir, "production_agent.pth")
        self.past_losses = []  # Track losses for adaptive training
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def _create_state(self, data_row):
        """Enhanced state representation with technical indicators"""
        try:
            # Basic price data
            basic_features = np.array([
                data_row['open'],
                data_row['high'],
                data_row['low'],
                data_row['close'],
                data_row['volume']
            ])

            # Add technical indicators if available
            tech_features = []
            for indicator in ['sma', 'rsi', 'macd', 'bbands']:
                if indicator in data_row:
                    tech_features.append(data_row[indicator])

            # Combine all features
            return np.concatenate([basic_features, tech_features])
            
        except Exception as e:
            self.logger.error(f"Error creating state: {str(e)}")
            return np.zeros(5)  # Return zero state as fallback

    def _calculate_reward(self, data_window, action: int) -> float:
        """Enhanced reward calculation with exploration bonus"""
        try:
            # Calculate base reward (PnL)
            price_change = (data_window['close'].iloc[-1] - data_window['close'].iloc[0]) 
            price_return = price_change / data_window['close'].iloc[0]
            
            # Calculate volatility adjustment
            volatility = data_window['close'].std() / data_window['close'].mean()
            volatility_penalty = -abs(volatility) * 0.5
            
            # Signal alignment reward
            signal_reward = self._calculate_signal_reward(data_window, action)
            
            # Exploration bonus based on state novelty
            novelty_bonus = self._calculate_state_novelty(data_window)
            
            # Combine components
            reward = (
                price_return * 1.0 +          # Base return
                volatility_penalty * 0.3 +    # Volatility penalty
                signal_reward * 0.4 +         # Signal alignment
                novelty_bonus * 0.2           # Exploration bonus
            )
            
            # Clip reward to prevent extreme values
            return float(np.clip(reward, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0

    def _calculate_signal_reward(self, data_window, action: int) -> float:
        """Calculate reward based on action alignment with technical signals"""
        try:
            # Technical signals
            sma_signal = 1 if data_window['close'].iloc[-1] > data_window['close'].mean() else -1
            momentum_signal = 1 if data_window['close'].diff().iloc[-1] > 0 else -1
            
            # Convert action to signal (-1 for sell, 0 for hold, 1 for buy)
            action_signal = action - 1 if action != 1 else 1
            
            # Calculate alignment
            signal_alignment = (
                (action_signal * sma_signal) + 
                (action_signal * momentum_signal)
            ) / 2
            
            return signal_alignment
            
        except Exception as e:
            self.logger.error(f"Error calculating signal reward: {str(e)}")
            return 0.0

    def _calculate_state_novelty(self, data_window) -> float:
        """Calculate state novelty for exploration bonus"""
        try:
            # Create state representation
            current_state = self._create_state(data_window.iloc[-1])
            
            # Calculate moving average of state values
            if not hasattr(self, 'state_history'):
                self.state_history = []
            
            self.state_history.append(current_state)
            self.state_history = self.state_history[-1000:]  # Keep last 1000 states
            
            if len(self.state_history) < 2:
                return 1.0
                
            # Calculate distance to previous states
            state_array = np.array(self.state_history[:-1])
            distances = np.linalg.norm(state_array - current_state, axis=1)
            min_distance = distances.min()
            
            # Convert to novelty score
            novelty = 1.0 - np.exp(-min_distance)
            return novelty
            
        except Exception as e:
            self.logger.error(f"Error calculating state novelty: {str(e)}")
            return 0.0

    def pre_train_agent(self, agent: DeepQAgent, data_manager, symbols: list, 
                       start_date: str, end_date: str):
        """Enhanced pre-training with adaptive parameters"""
        self.logger.info("Starting pre-training on historical data...")
        
        try:
            total_samples = 0
            running_loss = []
            
            for symbol in symbols:
                historical_data = data_manager.get_market_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )

                if historical_data.empty:
                    continue

                # Pre-process data with technical indicators
                historical_data = self._add_technical_indicators(historical_data)

                for i in range(len(historical_data) - 1):
                    # Create training example
                    state = self._create_state(historical_data.iloc[i])
                    next_state = self._create_state(historical_data.iloc[i + 1])
                    action = self._get_optimal_action(historical_data.iloc[i:i+2])
                    reward = self._calculate_reward(historical_data.iloc[i:i+2], action)
                    done = i == len(historical_data) - 2

                    # Add to replay buffer
                    agent.update_memory(state, action, reward, next_state, done)
                    total_samples += 1

                    # Adaptive training
                    if len(agent.memory) > agent.config.batch_size:
                        metrics = agent.train_step()
                        if metrics:
                            running_loss.append(metrics.get('loss', 0))
                            if len(running_loss) > 100:
                                running_loss.pop(0)
                            
                            # Adjust learning parameters
                            self._adjust_training_params(agent, running_loss)
                            
                            if i % 100 == 0:
                                self.logger.info(
                                    f"Pre-training progress - "
                                    f"Loss: {metrics.get('loss', 0):.4f}, "
                                    f"Epsilon: {agent.epsilon:.4f}, "
                                    f"Samples: {total_samples}"
                                )

            self.logger.info(f"Pre-training completed with {total_samples} samples")
            self.save_agent(agent)
            
        except Exception as e:
            self.logger.error(f"Error during pre-training: {e}")

    def _adjust_training_params(self, agent: DeepQAgent, running_loss: List[float]):
        """Adjust training parameters based on performance"""
        try:
            if len(running_loss) < 10:
                return
                
            loss_std = np.std(running_loss)
            loss_mean = np.mean(running_loss)
            
            # Adjust learning rate
            if loss_std > loss_mean * 0.5:  # High variance
                new_lr = max(agent.config.learning_rate * 0.95, 1e-6)
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
            # Adjust batch size
            if loss_std < loss_mean * 0.1:  # Low variance
                agent.config.batch_size = min(
                    agent.config.batch_size * 2, 
                    512
                )
            
            # Adjust epsilon decay
            if loss_mean < 0.1:  # Good performance
                agent.config.epsilon_decay = 0.999
            else:
                agent.config.epsilon_decay = 0.995
                
        except Exception as e:
            self.logger.error(f"Error adjusting training parameters: {str(e)}")

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to historical data"""
        try:
            df = data.copy()
            
            # Simple Moving Average
            df['sma'] = df['close'].rolling(window=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            
            # Bollinger Bands
            df['bbands_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bbands_upper'] = df['bbands_middle'] + (std * 2)
            df['bbands_lower'] = df['bbands_middle'] - (std * 2)
            
            return df.fillna(method='ffill')
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return data

    def get_agent(self, agent_config: DeepQAgentConfig) -> DeepQAgent:
        """Create a new agent or load existing one from saved state"""
        try:
            agent = DeepQAgent(agent_config)
            
            # Load saved model if it exists
            if os.path.exists(self.agent_path):
                self.logger.info("Loading existing agent from saved state")
                agent.load_model(self.agent_path)
            else:
                self.logger.info("Creating new agent")
                
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating/loading agent: {str(e)}")
            raise