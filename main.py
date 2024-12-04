# main.py
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import signal
import sys
import traceback

from config.trading_config import ConfigManager
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.agents.deep_q_agent import DeepQAgent, DeepQAgentConfig
from src.features.technical_indicators import TechnicalAnalyzer
from src.features.market_sentiment import MarketSentimentAnalyzer
from src.risk_management.portfolio_manager import PortfolioManager
from src.data.data_manager import DataManager
from src.models.agent_manager import AgentManager

class TradingSystem:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_api_config()
        
        self.logger.info("Initializing Trading System...")
        
        # Initialize components
        self.agent_manager = AgentManager(save_dir="models/saved_agents")
        
        # Modified initialize_components method
        self.initialize_components()

        # Trading state
        self.is_running = False
        self.last_update = None
        self.performance_metrics = {
            'trades': [],
            'portfolio_values': [],
            'returns': [],
            'positions': []
        }
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f"trading_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def initialize_components(self):
        """Initialize all trading system components."""
        try:
            # Initialize DataManager first since other components need it
            self.data_manager = DataManager(self.config)
            
            # Initialize analyzers
            self.sentiment_analyzer = SentimentAnalyzer(
                custom_config=self.config,
                data_manager=self.data_manager
            )
            self.market_sentiment = MarketSentimentAnalyzer(
                config=self.config,
                data_manager=self.data_manager
            )
            self.tech_analyzer = TechnicalAnalyzer(data_manager=self.data_manager)
            
            # Initialize portfolio manager
            self.portfolio_manager = PortfolioManager(self.config, data_manager=self.data_manager)
            
            # Initialize trading agent with persistence
            state_dim = self._calculate_state_dim()
            agent_config = DeepQAgentConfig(
                state_dim=state_dim,
                action_dim=3,  # Buy, Hold, Sell
                hidden_dims=[512, 256, 128],
                learning_rate=0.001,
                gamma=0.99,
                initial_epsilon=0.5,  # Lower initial epsilon for pretrained agent
                final_epsilon=0.01,
                epsilon_decay=0.995
            )
            
            # Get or create agent
            self.agent = self.agent_manager.get_agent(agent_config)
            
            # Pre-train if new agent
            if self.agent.steps == 0:
                self.logger.info("New agent detected. Starting pre-training...")
                self._pre_train_agent()
            else:
                self.logger.info(f"Loaded existing agent with {self.agent.steps} training steps")
            
            # Load trading universe
            self.symbols = self.config_manager.get('TRADING_SYMBOLS', 
                                                ['AAPL', 'MSFT', 'GOOGL'])
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
            
    def _pre_train_agent(self):
        """Pre-train agent on historical data"""
        try:
            # Get historical data for pre-training
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # One year of data
            
            total_samples = 0
            for symbol in self.symbols:
                self.logger.info(f"Pre-training on {symbol}")
                
                # Get historical data
                data = self.data_manager.get_market_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data.empty:
                    self.logger.warning(f"No historical data available for {symbol}")
                    continue
                
                # Create training examples
                for i in range(len(data) - 1):
                    # Get state and next state
                    state = self._prepare_agent_state({
                        'technical': self.tech_analyzer.calculate_features(data.iloc[:i+1]),
                        'sentiment': 0,  # Historical sentiment not available
                        'portfolio': self.portfolio_manager.get_portfolio_metrics()
                    })
                    
                    next_state = self._prepare_agent_state({
                        'technical': self.tech_analyzer.calculate_features(data.iloc[:i+2]),
                        'sentiment': 0,
                        'portfolio': self.portfolio_manager.get_portfolio_metrics()
                    })
                    
                    # Calculate reward (simple price change for now)
                    price_change = (data.iloc[i+1]['close'] - data.iloc[i]['close']) / data.iloc[i]['close']
                    reward = price_change * 100  # Scale reward for better learning
                    
                    # Determine optimal action (simplified)
                    action = 0 if price_change > 0.01 else (2 if price_change < -0.01 else 1)
                    
                    # Add to replay buffer
                    self.agent.update_memory(state, action, reward, next_state, False)
                    total_samples += 1
                    
                    # Train periodically
                    if total_samples % 100 == 0:
                        metrics = self.agent.train_step()
                        if metrics:
                            self.logger.info(f"Pre-training metrics: {metrics}")
                            
            self.logger.info(f"Pre-training completed with {total_samples} samples")
            
            # Save pre-trained agent
            self.agent_manager.save_agent(self.agent)
            
        except Exception as e:
            self.logger.error(f"Error during pre-training: {e}")
            
    async def run(self):
        """Main trading loop"""
        self.is_running = True
        self.logger.info("Starting trading system...")
        
        try:
            while self.is_running:
                await self.trading_loop()
                
        except Exception as e:
            self.logger.error(f"Error in main trading loop: {str(e)}")
            self.logger.error(traceback.format_exc())
            await self.shutdown()
        finally:
            # Save agent state before exiting
            self.agent_manager.save_agent(self.agent)
            
    async def trading_loop(self):
        """Single iteration of the trading loop"""
        try:
            current_time = datetime.now()
            
            # Check if market is open
            if not self.is_market_open():
                wait_time = self.time_to_next_market_open()
                self.logger.info(f"Market closed. Waiting for {wait_time}")
                await asyncio.sleep(wait_time.total_seconds())
                return
                
            # Update market state
            state = await self.update_market_state()
            
            # Make trading decisions
            await self.make_trading_decisions(state)
            
            # Update and log metrics
            await self.update_metrics()
            
            # Train agent
            await self.train_agent()
            
            # Save state periodically
            await self.periodic_save()
            
            # Calculate wait time for next iteration
            wait_time = self.calculate_wait_time()
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    async def update_market_state(self) -> Dict:
        """Update and return current market state."""
        self.logger.info("Updating market state...")
        state = {}

        try:
            start_date = datetime.now() - timedelta(days=30)  # Fetch last 30 days
            end_date = datetime.now()

            for symbol in self.symbols:
                # Get sentiment analysis
                sentiment = await self.market_sentiment.analyze_sentiment(symbol)

                # Get technical features
                market_data = self._get_market_data(symbol, start_date, end_date)
                technical = self.tech_analyzer.calculate_features(
                    data=market_data, start_date=start_date, end_date=end_date
                )

                # Get portfolio state
                portfolio = await self.portfolio_manager.update_portfolio()

                state[symbol] = {
                    'sentiment': sentiment,
                    'technical': technical,
                    'portfolio': portfolio,
                    'timestamp': datetime.now()
                }

            self.last_update = datetime.now()
            return state

        except Exception as e:
            self.logger.error(f"Error updating market state: {str(e)}")
            return {}
            
    async def make_trading_decisions(self, state: Dict):
        """Make and execute trading decisions"""
        self.logger.info("Making trading decisions...")
        
        try:
            for symbol in self.symbols:
                if symbol not in state:
                    continue
                    
                # Prepare state for agent
                agent_state = self._prepare_agent_state(state[symbol])
                
                # Get action from agent
                action = self.agent.select_action(agent_state)
                
                # Execute action
                await self.execute_action(symbol, action, state[symbol])
                
        except Exception as e:
            self.logger.error(f"Error making trading decisions: {str(e)}")
            
    async def execute_action(self, symbol: str, action: int, state: Dict):
        """Execute trading action"""
        try:
            if action == 1:  # Hold
                return
                
            # Calculate position size
            position_size = self.calculate_position_size(symbol, state)
            
            # Execute trade
            if action == 0:  # Buy
                await self.portfolio_manager.open_position(symbol, position_size)
                self.logger.info(f"Opened long position in {symbol}: {position_size}")
            elif action == 2:  # Sell
                await self.portfolio_manager.close_position(symbol)
                self.logger.info(f"Closed position in {symbol}")
                
            # Record trade
            self.performance_metrics['trades'].append({
                'symbol': symbol,
                'action': ['buy', 'hold', 'sell'][action],
                'size': position_size,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error executing action for {symbol}: {str(e)}")
            
    def calculate_position_size(self, symbol: str, state: Dict) -> float:
        """Calculate position size based on risk management rules"""
        try:
            portfolio = state['portfolio']
            
            # Get maximum position size from risk limits
            max_position = portfolio.total_value * self.config_manager.get('MAX_POSITION_SIZE', 0.1)
            
            # Adjust based on conviction
            conviction = (state['sentiment'].composite_score + 1) / 2  # Normalize to [0, 1]
            
            position_size = max_position * conviction
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    async def train_agent(self):
        """Train the trading agent"""
        try:
            if len(self.agent.memory) >= self.agent.config.batch_size:
                metrics = self.agent.train_step()
                self.logger.debug(f"Agent training metrics: {metrics}")
                
        except Exception as e:
            self.logger.error(f"Error training agent: {str(e)}")
            
    async def update_metrics(self):
        """Update and log performance metrics."""
        try:
            portfolio = await self.portfolio_manager.update_portfolio()

            self.performance_metrics['portfolio_values'].append(portfolio.total_value)

            # Add returns or placeholder value
            if len(self.performance_metrics['portfolio_values']) > 1:
                returns = (
                    portfolio.total_value /
                    self.performance_metrics['portfolio_values'][-2] - 1
                )
            else:
                returns = 0.0  # No prior value for returns

            self.performance_metrics['returns'].append(returns)

            # Log current metrics
            self.log_metrics()

        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def log_metrics(self):
        """Log current performance metrics"""
        try:
            if self.performance_metrics['portfolio_values']:
                current_value = self.performance_metrics['portfolio_values'][-1]
                initial_value = self.performance_metrics['portfolio_values'][0]
                total_return = (current_value / initial_value - 1) * 100
                
                self.logger.info(f"""
                Performance Metrics:
                Current Portfolio Value: ${current_value:,.2f}
                Total Return: {total_return:.2f}%
                Number of Trades: {len(self.performance_metrics['trades'])}
                """)
                
        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")
            
    async def periodic_save(self):
        """Periodically save system state"""
        try:
            current_time = datetime.now()
            save_interval = timedelta(hours=1)
            
            if (not hasattr(self, 'last_save') or 
                current_time - self.last_save > save_interval):
                
                # Save agent state
                self.agent.save(f"agent_state_{current_time.strftime('%Y%m%d_%H%M%S')}.pth")
                
                # Save metrics
                self.save_metrics()
                
                self.last_save = current_time
                self.logger.info("System state saved")
                
        except Exception as e:
            self.logger.error(f"Error saving system state: {str(e)}")
            
    def save_metrics(self):
        """Save performance metrics to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Align metrics before saving
            max_len = max(len(self.performance_metrics['portfolio_values']), len(self.performance_metrics['returns']))
            self.performance_metrics['portfolio_values'].extend([None] * (max_len - len(self.performance_metrics['portfolio_values'])))
            self.performance_metrics['returns'].extend([None] * (max_len - len(self.performance_metrics['returns'])))

            # Save trades
            pd.DataFrame(self.performance_metrics['trades']).to_csv(f"trades_{timestamp}.csv", index=False)

            # Save performance metrics
            pd.DataFrame({
                'portfolio_value': self.performance_metrics['portfolio_values'],
                'returns': self.performance_metrics['returns']
            }).to_csv(f"performance_{timestamp}.csv", index=False)

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            
    def is_market_open(self) -> bool:
        """Check if market is open"""
        # Implement market hours check
        return True
        
    def time_to_next_market_open(self) -> timedelta:
        """Calculate time until next market open"""
        # Implement market opening time calculation
        return timedelta(hours=12)
        
    def calculate_wait_time(self) -> float:
        """Calculate wait time until next iteration"""
        return 60  # 1 minute
        
    def _calculate_state_dim(self) -> int:
        """Calculate state dimension for agent"""
        return 10  # Adjust based on features used
        
    def _prepare_agent_state(self, state: Dict) -> np.ndarray:
        """Prepare state vector for agent"""
        # Combine features into state vector
        features = []
        
        # Add sentiment features
        features.append(state['sentiment'].composite_score)
        
        # Add technical features
        tech = state['technical']
        features.extend([
            tech.trend_indicators['sma_short'] / tech.trend_indicators['sma_long'],
            (tech.momentum_indicators['rsi'] - 50) / 50,
            tech.trend_indicators['macd'] / 100,
            tech.volatility_indicators['historical_volatility'],
            tech.volume_indicators['volume_ma_ratio'] - 1
        ])
        
        # Add portfolio features
        portfolio = state['portfolio']
        features.extend([
            portfolio.total_value,
            portfolio.cash / portfolio.total_value,
            portfolio.leverage
        ])
        
        return np.array(features)
        
    def _get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch market data using DataManager"""
        try:
            data = self.data_manager.get_market_data(symbol, start_date=start_date, end_date=end_date)
            if data.empty:
                self.logger.warning(f"No market data available for {symbol}.")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return pd.DataFrame()
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signal"""
        self.logger.info("Shutdown signal received")
        asyncio.create_task(self.shutdown())
        
    async def shutdown(self):
        """Shutdown the trading system"""
        self.logger.info("Shutting down trading system...")
        self.is_running = False
        
        try:
            # Close all positions if configured
            if self.config_manager.get('CLOSE_POSITIONS_ON_SHUTDOWN', True):
                await self.portfolio_manager.close_all_positions()
                
            # Save final state with agent manager
            self.agent_manager.save_agent(self.agent)
            self.save_metrics()
            
            self.logger.info("Trading system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
        finally:
            sys.exit(0)

def main():
    """Main entry point"""
    trading_system = TradingSystem()
    asyncio.run(trading_system.run())

if __name__ == "__main__":
    main()