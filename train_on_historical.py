import asyncio
import logging
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np

from config.trading_config import ConfigManager
from src.agents.deep_q_agent import DeepQAgent, DeepQAgentConfig
from src.models.agent_manager import AgentManager
from src.data.data_manager import DataManager
from src.environment import TradingEnvironment
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def train_agent_historical(
    symbols: List[str],
    start_date: str,
    end_date: str,
    episodes: int = 100,
    steps_per_episode: int = 1000
):
    """Train agent on historical data"""
    try:
        # Initialize components
        config_manager = ConfigManager()
        config = config_manager.get_api_config()
        
        # Create data config
        data_config = {
            'alpaca_api_key': config['alpaca_api_key'],
            'alpaca_secret_key': config['alpaca_secret_key'],
            'data_config': {
                'cache_duration': 15,  # minutes
                'db_path': 'market_data.db',
                'start_date': start_date,
                'end_date': end_date
            }
        }
        
        # Initialize clients
        trading_client = TradingClient(
            config['alpaca_api_key'],
            config['alpaca_secret_key']
        )
        data_client = StockHistoricalDataClient(
            config['alpaca_api_key'],
            config['alpaca_secret_key']
        )
        
        # Initialize data manager with config
        data_manager = DataManager(config=data_config)
        
        # Set up environment config
        env_config = {
            'max_steps': steps_per_episode,
            'initial_balance': 41000,
            'trading_fee': 0.001,
            'symbols': symbols,
            'data_config': data_config,
            'symbol': symbols[0],  # Primary symbol for training
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Initialize environment
        env = TradingEnvironment(
            trading_client=trading_client,
            data_client=data_client,
            config=env_config
        )

        # Set up agent
        state_dim = 50  # Adjust based on your feature set
        action_dim = 3  # Buy, Hold, Sell
        
        agent_config = DeepQAgentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[512, 256, 128],
            learning_rate=0.0001,
            gamma=0.99,
            initial_epsilon=0.9,
            final_epsilon=0.05,
            epsilon_decay=0.995
        )

        # Initialize agent manager
        agent_manager = AgentManager()
        agent = agent_manager.get_agent(agent_config)

        logger.info("Starting training on historical data...")
        logger.info(f"Training on symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        best_reward = float('-inf')
        episode_rewards = []

        for episode in range(episodes):
            total_reward = 0
            state = await env.reset()
            
            for step in range(steps_per_episode):
                # Select action
                action = agent.select_action(env.state_normalizer.normalize(state))
                
                # Take action in environment
                next_state, reward, done, info = await env.step(action)
                
                # Store transition in replay buffer
                agent.update_memory(
                    env.state_normalizer.normalize(state),
                    action,
                    reward,
                    env.state_normalizer.normalize(next_state),
                    done
                )
                
                # Train agent
                if len(agent.memory) > agent.config.batch_size:
                    metrics = agent.train_step()
                    if step % 100 == 0:
                        logger.info(f"Episode {episode + 1}, Step {step + 1}, Loss: {metrics.get('loss', 0):.4f}")
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            logger.info(f"""
            Episode {episode + 1} completed:
            Total Reward: {total_reward:.2f}
            Average Reward (last 100): {avg_reward:.2f}
            Epsilon: {agent.epsilon:.3f}
            """)
            
            # Save if best performing
            if total_reward > best_reward:
                best_reward = total_reward
                agent_manager.save_agent(agent)
                logger.info(f"New best reward: {best_reward:.2f}, Agent saved")
            
            # Save training metrics
            pd.DataFrame({
                'episode': range(len(episode_rewards)),
                'reward': episode_rewards
            }).to_csv(f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        logger.info("Training completed!")
        return agent
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

async def main():
    symbols = ['AAPL', 'SPY', 'NVDA', 'QQQ', 'TSLA']
    start_date = '2022-12-01'
    end_date = '2023-12-31'  # Changed to past date for historical data
    
    await train_agent_historical(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        episodes=100,
        steps_per_episode=1000
    )

if __name__ == "__main__":
    asyncio.run(main())