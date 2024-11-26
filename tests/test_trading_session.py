# tests/test_trading_session.py
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List

from src.utils.config import ConfigManager
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.agents.deep_q_agent import DeepQAgent
from src.features.technical_indicators import TechnicalAnalyzer
from src.risk_management.portfolio_manager import PortfolioManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestTradingSession:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.test_duration = timedelta(minutes=30)
        
        # Initialize components
        self.setup_components()
        
        # Trading metrics
        self.trades = []
        self.performance_metrics = {
            'returns': [],
            'positions': [],
            'portfolio_values': []
        }
        
    def setup_components(self):
        """Initialize all trading components"""
        config = self.config_manager.get_api_config()
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(config)
        
        # Initialize agent
        state_dim = self._calculate_state_dim()
        self.agent = DeepQAgent(
            state_dim=state_dim,
            action_dim=3  # buy, hold, sell
        )
        
    async def run_session(self):
        """Run a test trading session"""
        logger.info("Starting test trading session...")
        
        start_time = datetime.now()
        iteration = 0
        
        try:
            while datetime.now() - start_time < self.test_duration:
                iteration += 1
                logger.info(f"\nIteration {iteration}")
                
                # Update market state
                await self.update_market_state()
                
                # Make trading decisions
                await self.execute_trading_decisions()
                
                # Update metrics
                await self.update_metrics()
                
                # Log progress
                self.log_progress()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1-minute intervals
                
            # Session complete - show results
            self.show_results()
            
        except Exception as e:
            logger.error(f"Trading session failed: {str(e)}")
            raise
        finally:
            self.cleanup()
            
    async def update_market_state(self):
        """Update market state for all symbols"""
        self.current_state = {}
        
        for symbol in self.symbols:
            # Get sentiment
            sentiment = await self.sentiment_analyzer.analyze_market_sentiment(symbol)
            
            # Get technical features
            tech_features = self.tech_analyzer.calculate_features(
                self._get_recent_data(symbol)
            )
            
            # Combine into state
            self.current_state[symbol] = {
                'sentiment': sentiment['combined'],
                'technical': tech_features,
                'timestamp': datetime.now()
            }
            
            logger.info(f"{symbol} - Sentiment: {sentiment['combined']:.2f}")
            
    async def execute_trading_decisions(self):
        """Execute trading decisions based on current state"""
        for symbol in self.symbols:
            # Prepare state for agent
            state = self._prepare_state(symbol)
            
            # Get action from agent
            action = self.agent.select_action(state)
            
            # Execute action
            try:
                if action != 1:  # If not hold
                    position_size = self._calculate_position_size(symbol)
                    await self.portfolio_manager.update_position(
                        symbol,
                        position_size if action == 0 else -position_size
                    )
                    
                    self.trades.append({
                        'symbol': symbol,
                        'action': ['buy', 'hold', 'sell'][action],
                        'size': position_size,
                        'timestamp': datetime.now()
                    })
                    
                    logger.info(f"Executed {['buy', 'hold', 'sell'][action]} for {symbol}")
                    
            except Exception as e:
                logger.error(f"Trade execution failed for {symbol}: {str(e)}")
                
    async def update_metrics(self):
        """Update performance metrics"""
        # Get portfolio state
        portfolio = await self.portfolio_manager.update_portfolio()
        
        # Update metrics
        self.performance_metrics['portfolio_values'].append(portfolio.total_value)
        if len(self.performance_metrics['portfolio_values']) > 1:
            returns = (portfolio.total_value / 
                      self.performance_metrics['portfolio_values'][-2] - 1)
            self.performance_metrics['returns'].append(returns)
            
        self.performance_metrics['positions'].append(
            {symbol: pos for symbol, pos in portfolio.positions.items()}
        )
        
    def log_progress(self):
        """Log current session progress"""
        if self.performance_metrics['portfolio_values']:
            current_value = self.performance_metrics['portfolio_values'][-1]
            initial_value = self.performance_metrics['portfolio_values'][0]
            total_return = (current_value / initial_value - 1) * 100
            
            logger.info(f"Current Portfolio Value: ${current_value:,.2f}")
            logger.info(f"Total Return: {total_return:.2f}%")
            logger.info(f"Number of Trades: {len(self.trades)}")
            
    def show_results(self):
        """Show final session results"""
        logger.info("\n=== Test Session Results ===")
        
        # Calculate metrics
        initial_value = self.performance_metrics['portfolio_values'][0]
        final_value = self.performance_metrics['portfolio_values'][-1]
        total_return = (final_value / initial_value - 1) * 100
        
        returns = pd.Series(self.performance_metrics['returns'])
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Print results
        logger.info(f"Session Duration: {self.test_duration}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Number of Trades: {len(self.trades)}")
        logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save session results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades
        pd.DataFrame(self.trades).to_csv(f"test_trades_{timestamp}.csv")
        
        # Save metrics
        pd.DataFrame({
            'portfolio_value': self.performance_metrics['portfolio_values'],
            'returns': self.performance_metrics['returns']
        }).to_csv(f"test_metrics_{timestamp}.csv")
        
        logger.info("Results saved to CSV files")
        
    def cleanup(self):
        """Clean up resources"""
        # Save agent if needed
        self.agent.save(f"agent_test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        
    def _calculate_state_dim(self) -> int:
        """Calculate state dimension"""
        # Example: sentiment + 5 technical features + position info
        return 7
        
    def _prepare_state(self, symbol: str) -> np.ndarray:
        """Prepare state vector for agent"""
        state = np.zeros(self._calculate_state_dim())
        
        # Fill state vector with features
        state[0] = self.current_state[symbol]['sentiment']
        state[1:6] = self._get_technical_features(symbol)
        state[6] = self._get_position_size(symbol)
        
        return state
        
    def _get_recent_data(self, symbol: str) -> pd.DataFrame:
        """Get recent market data"""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return ticker.history(period="1mo")
        
    def _calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on portfolio value and risk limits"""
        return 100  # Simplified for testing
        
    def _get_technical_features(self, symbol: str) -> np.ndarray:
            """Extract key technical features"""
            features = self.current_state[symbol]['technical']
            
            # Extract key indicators and normalize them
            normalized_features = np.array([
                features.trend_indicators['sma_short'] / features.trend_indicators['sma_long'],
                (features.momentum_indicators['rsi'] - 50) / 50,  # Normalize RSI to [-1, 1]
                features.momentum_indicators['macd'] / 100,  # Normalize MACD
                min(features.volatility_indicators['historical_volatility'], 1.0),  # Cap volatility
                features.volume_indicators['volume_ma_ratio'] - 1  # Center volume ratio around 0
            ])
            
            return normalized_features
            
    def _get_position_size(self, symbol: str) -> float:
        """Get current position size"""
        try:
            portfolio = self.portfolio_manager.positions.get(symbol, 0)
            return portfolio
        except Exception as e:
            logger.error(f"Error getting position size for {symbol}: {str(e)}")
            return 0.0

def main():
    """Run test trading session"""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        
        # Create and run session
        session = TestTradingSession()
        asyncio.run(session.run_session())
        
    except Exception as e:
        logger.error(f"Test session failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()